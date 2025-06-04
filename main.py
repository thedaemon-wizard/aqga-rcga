import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer.primitives import SamplerV2
from qiskit_aer import AerSimulator
import time
from typing import Callable, Tuple, List, Dict
import os
from datetime import datetime

# resultsフォルダの作成
os.makedirs('results', exist_ok=True)

# ===== ベンチマーク関数の定義 =====
class BenchmarkFunctions:
    """実数値連続最適化問題のベンチマーク関数群"""
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Sphere関数 (単峰性)"""
        return np.sum(x**2)
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Rosenbrock関数 (谷型)"""
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin関数 (多峰性)"""
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Ackley関数 (多峰性)"""
        n = len(x)
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - \
               np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e
    
    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """Griewank関数 (多峰性)"""
        return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    
    @staticmethod
    def schwefel(x: np.ndarray) -> float:
        """Schwefel関数 (多峰性、非対称)"""
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# テスト関数の設定
TEST_FUNCTIONS = {
    'sphere': {
        'func': BenchmarkFunctions.sphere,
        'bounds': (-5.12, 5.12),
        'optimum': 0.0,
        'dim': 10
    },
    'rosenbrock': {
        'func': BenchmarkFunctions.rosenbrock,
        'bounds': (-2.048, 2.048),
        'optimum': 0.0,
        'dim': 10
    },
    'rastrigin': {
        'func': BenchmarkFunctions.rastrigin,
        'bounds': (-5.12, 5.12),
        'optimum': 0.0,
        'dim': 10
    },
    'ackley': {
        'func': BenchmarkFunctions.ackley,
        'bounds': (-32.768, 32.768),
        'optimum': 0.0,
        'dim': 10
    },
    'griewank': {
        'func': BenchmarkFunctions.griewank,
        'bounds': (-600, 600),
        'optimum': 0.0,
        'dim': 10
    },
    'schwefel': {
        'func': BenchmarkFunctions.schwefel,
        'bounds': (-500, 500),
        'optimum': 0.0,
        'dim': 10
    }
}

# ===== RCGA (Real-Coded Genetic Algorithm) の実装 =====
class RCGA:
    """実数値遺伝的アルゴリズム"""
    
    def __init__(self, 
                 pop_size: int = 50,
                 dim: int = 10,
                 bounds: Tuple[float, float] = (-5.12, 5.12),
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 mutation_scale: float = 0.1):
        self.pop_size = pop_size
        self.dim = dim
        self.bounds = bounds
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        
    def initialize_population(self) -> np.ndarray:
        """初期個体群の生成"""
        return np.random.uniform(
            self.bounds[0], self.bounds[1], 
            size=(self.pop_size, self.dim)
        )
    
    def selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """トーナメント選択"""
        selected = np.empty_like(population)
        for i in range(self.pop_size):
            idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
            selected[i] = population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]
        return selected
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """BLX-α交叉"""
        if np.random.rand() < self.crossover_rate:
            alpha = 0.5
            d = np.abs(parent1 - parent2)
            min_vals = np.minimum(parent1, parent2) - alpha * d
            max_vals = np.maximum(parent1, parent2) + alpha * d
            
            child1 = np.random.uniform(min_vals, max_vals)
            child2 = np.random.uniform(min_vals, max_vals)
            
            # 境界チェック
            child1 = np.clip(child1, self.bounds[0], self.bounds[1])
            child2 = np.clip(child2, self.bounds[0], self.bounds[1])
            
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutation(self, individual: np.ndarray) -> np.ndarray:
        """ガウス突然変異"""
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.normal(0, self.mutation_scale * (self.bounds[1] - self.bounds[0]))
                individual[i] = np.clip(individual[i], self.bounds[0], self.bounds[1])
        return individual
    
    def evolve(self, objective_func: Callable, max_iter: int = 100) -> Dict:
        """進化プロセス"""
        population = self.initialize_population()
        
        best_fitness_history = []
        avg_fitness_history = []
        best_individual = None
        best_fitness = float('inf')
        
        for generation in range(max_iter):
            # 適応度評価
            fitness = np.array([objective_func(ind) for ind in population])
            
            # 最良個体の更新
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_individual = population[min_idx].copy()
            
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(fitness))
            
            # 選択
            selected = self.selection(population, fitness)
            
            # 交叉と突然変異
            new_population = []
            for i in range(0, self.pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[min(i + 1, self.pop_size - 1)]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = np.array(new_population[:self.pop_size])
            
            # エリート保存
            population[0] = best_individual
        
        return {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history
        }

# ===== 改善されたAQGA (Adaptive Quantum Genetic Algorithm) の実装 =====
class ImprovedAQGA:
    """改善された適応的量子遺伝的アルゴリズム (Qiskit 2.0 SamplerV2対応)"""
    
    def __init__(self,
                 pop_size: int = 30,  # 個体数を増加
                 dim: int = 10,
                 n_qubits_per_dim: int = 10,  # 精度を向上
                 bounds: Tuple[float, float] = (-5.12, 5.12),
                 theta_max: float = 0.25 * np.pi,  # より大きな初期回転角
                 theta_min: float = 0.01 * np.pi,  # より小さな最終回転角
                 mutation_rate: float = 0.1,  # 突然変異率を増加
                 disaster_rate: float = 0.3,
                 disaster_threshold: int = 5,  # より早い災害発動
                 local_search_rate: float = 0.2,  # ローカルサーチの確率
                 shots: int = 100):  # 測定回数を増加
        self.pop_size = pop_size
        self.dim = dim
        self.n_qubits_per_dim = n_qubits_per_dim
        self.bounds = bounds
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.mutation_rate = mutation_rate
        self.disaster_rate = disaster_rate
        self.disaster_threshold = disaster_threshold
        self.local_search_rate = local_search_rate
        self.shots = shots
        
        # Qiskit 2.0のSamplerV2
        self.backend = AerSimulator()
        self.sampler = SamplerV2()
        
        # 適応的パラメータ
        self.adaptive_mutation_rate = mutation_rate
        self.adaptive_theta_scale = 1.0
    
    def create_quantum_circuit_for_dim(self) -> QuantumCircuit:
        """単一次元用の量子回路を作成"""
        qr = QuantumRegister(self.n_qubits_per_dim, 'q')
        cr = ClassicalRegister(self.n_qubits_per_dim, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # 初期状態：全量子ビットを重ね合わせ状態に
        for i in range(self.n_qubits_per_dim):
            qc.h(i)
        
        return qc
    
    def create_quantum_individual(self) -> List[QuantumCircuit]:
        """量子個体（各次元の量子回路のリスト）の作成"""
        return [self.create_quantum_circuit_for_dim() for _ in range(self.dim)]
    
    def measure_quantum_circuits_batch(self, circuits: List[QuantumCircuit]) -> List[str]:
        """複数の量子回路をバッチで測定（SamplerV2使用）"""
        # 測定を追加した回路のコピーを作成
        measured_circuits = []
        for qc in circuits:
            qc_copy = qc.copy()
            qc_copy.measure_all()
            measured_circuits.append(qc_copy)
        
        # SamplerV2でバッチ実行（複数回測定）
        job = self.sampler.run(measured_circuits, shots=self.shots)
        results = job.result()
        
        # 結果の取得（最頻値を使用）
        binary_strings = []
        for i, circuit in enumerate(measured_circuits):
            counts = results[i].data.meas.get_counts()
            # 最も頻度の高い測定結果を取得
            max_key = max(counts.keys(), key=counts.get)
            # ビット順序を調整（Qiskitは逆順で返すため）
            binary_str = max_key[::-1]
            binary_strings.append(binary_str)
        
        return binary_strings
    
    def measure_quantum_individual(self, individual: List[QuantumCircuit]) -> List[str]:
        """量子個体全体の測定（各次元をバッチで測定）"""
        return self.measure_quantum_circuits_batch(individual)
    
    def decode_binary_to_real(self, binary_list: List[str]) -> np.ndarray:
        """バイナリ文字列のリストを実数値ベクトルにデコード（改善版）"""
        real_vector = np.zeros(self.dim)
        
        for i in range(self.dim):
            # バイナリを[0, 1]の実数に変換
            decimal_val = int(binary_list[i], 2)
            normalized_val = decimal_val / (2**self.n_qubits_per_dim - 1)
            
            # 指定された範囲にスケーリング
            real_vector[i] = self.bounds[0] + normalized_val * (self.bounds[1] - self.bounds[0])
        
        return real_vector
    
    def calculate_rotation_direction(self, current_bit: str, best_bit: str, 
                                   current_fitness: float, best_fitness: float) -> float:
        """改善された回転方向の計算（連続値を返す）"""
        if current_fitness < best_fitness:
            return 0.0
        
        # フィットネスの差に基づいて回転強度を調整
        fitness_diff = abs(current_fitness - best_fitness) / (abs(best_fitness) + 1e-10)
        rotation_strength = min(1.0, fitness_diff)
        
        if current_bit == '0' and best_bit == '1':
            return rotation_strength
        elif current_bit == '1' and best_bit == '0':
            return -rotation_strength
        else:
            # 同じビットでも小さな摂動を加える
            return 0.1 * rotation_strength * (2 * np.random.rand() - 1)
    
    def update_quantum_circuit(self, qc: QuantumCircuit, current_binary: str, 
                             best_binary: str, current_fitness: float, 
                             best_fitness: float, theta: float) -> QuantumCircuit:
        """単一の量子回路を更新"""
        for i in range(self.n_qubits_per_dim):
            direction = self.calculate_rotation_direction(
                current_binary[i], best_binary[i], current_fitness, best_fitness
            )
            
            if abs(direction) > 0:
                qc.ry(direction * theta * self.adaptive_theta_scale, i)
        
        return qc
    
    def update_quantum_individual(self, individual: List[QuantumCircuit], 
                                current_binary_list: List[str], 
                                best_binary_list: List[str], 
                                current_fitness: float, 
                                best_fitness: float, 
                                generation: int, 
                                max_iter: int) -> List[QuantumCircuit]:
        """量子個体全体の更新"""
        # 適応的回転角度の計算（非線形減衰）
        progress = generation / max_iter
        theta = self.theta_min + (self.theta_max - self.theta_min) * np.exp(-5 * progress)
        
        updated_individual = []
        for i in range(self.dim):
            qc_updated = individual[i].copy()
            qc_updated = self.update_quantum_circuit(
                qc_updated, current_binary_list[i], best_binary_list[i],
                current_fitness, best_fitness, theta
            )
            updated_individual.append(qc_updated)
        
        return updated_individual
    
    def apply_mutation(self, individual: List[QuantumCircuit]) -> List[QuantumCircuit]:
        """適応的量子突然変異"""
        mutated_individual = []
        for qc in individual:
            qc_mutated = qc.copy()
            for i in range(self.n_qubits_per_dim):
                if np.random.rand() < self.adaptive_mutation_rate:
                    # ランダムな回転を加える
                    angle = np.random.uniform(-np.pi/4, np.pi/4)
                    qc_mutated.ry(angle, i)
            mutated_individual.append(qc_mutated)
        return mutated_individual
    
    def apply_disaster(self, quantum_population: List[List[QuantumCircuit]]) -> List[List[QuantumCircuit]]:
        """災害演算子の適用（改善版）"""
        n_disaster = int(self.pop_size * self.disaster_rate)
        disaster_indices = np.random.choice(self.pop_size, n_disaster, replace=False)
        
        for idx in disaster_indices:
            if np.random.rand() < 0.5:
                # 完全に新しい個体を生成
                quantum_population[idx] = self.create_quantum_individual()
            else:
                # 部分的なリセット（半分の次元のみ）
                reset_dims = np.random.choice(self.dim, self.dim // 2, replace=False)
                for dim_idx in reset_dims:
                    quantum_population[idx][dim_idx] = self.create_quantum_circuit_for_dim()
        
        return quantum_population
    
    def local_search(self, individual: np.ndarray, objective_func: Callable, 
                    iterations: int = 10) -> np.ndarray:
        """簡単なローカルサーチ"""
        best = individual.copy()
        best_fitness = objective_func(best)
        
        step_size = 0.01 * (self.bounds[1] - self.bounds[0])
        
        for _ in range(iterations):
            # ランダムな次元を選択
            dim = np.random.randint(self.dim)
            
            # 両方向を試す
            for direction in [-1, 1]:
                candidate = best.copy()
                candidate[dim] += direction * step_size
                candidate[dim] = np.clip(candidate[dim], self.bounds[0], self.bounds[1])
                
                fitness = objective_func(candidate)
                if fitness < best_fitness:
                    best = candidate
                    best_fitness = fitness
        
        return best
    
    def update_adaptive_parameters(self, generation: int, max_iter: int, 
                                 improvement_rate: float):
        """適応的パラメータの更新"""
        progress = generation / max_iter
        
        # 改善率に基づいて突然変異率を調整
        if improvement_rate < 0.01:  # 改善が停滞
            self.adaptive_mutation_rate = min(0.3, self.mutation_rate * (1 + progress))
            self.adaptive_theta_scale = min(1.5, 1.0 + 0.5 * progress)
        else:
            self.adaptive_mutation_rate = self.mutation_rate
            self.adaptive_theta_scale = 1.0
    
    def evolve(self, objective_func: Callable, max_iter: int = 100) -> Dict:
        """進化プロセス"""
        # 初期量子個体群の生成
        quantum_population = [self.create_quantum_individual() for _ in range(self.pop_size)]
        
        best_fitness_history = []
        avg_fitness_history = []
        best_individual = None
        best_fitness = float('inf')
        best_binary = None
        stagnation_counter = 0
        prev_best_fitness = float('inf')
        
        for generation in range(max_iter):
            # 測定と評価
            binary_population = []
            real_population = []
            fitness_values = []
            
            # バッチ処理のために全個体の全回路を収集
            all_circuits = []
            for individual in quantum_population:
                all_circuits.extend(individual)
            
            # バッチ測定
            if all_circuits:
                all_measurements = self.measure_quantum_circuits_batch(all_circuits)
                
                # 結果を個体ごとに分割
                idx = 0
                for i in range(self.pop_size):
                    individual_measurements = all_measurements[idx:idx + self.dim]
                    idx += self.dim
                    
                    real_vector = self.decode_binary_to_real(individual_measurements)
                    
                    # ローカルサーチの適用
                    if np.random.rand() < self.local_search_rate:
                        real_vector = self.local_search(real_vector, objective_func)
                    
                    fitness = objective_func(real_vector)
                    
                    binary_population.append(individual_measurements)
                    real_population.append(real_vector)
                    fitness_values.append(fitness)
            
            # 最良個体の更新
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < best_fitness:
                best_fitness = fitness_values[min_idx]
                best_individual = real_population[min_idx].copy()
                best_binary = binary_population[min_idx]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(fitness_values))
            
            # 改善率の計算
            improvement_rate = abs(prev_best_fitness - best_fitness) / (abs(prev_best_fitness) + 1e-10)
            prev_best_fitness = best_fitness
            
            # 適応的パラメータの更新
            self.update_adaptive_parameters(generation, max_iter, improvement_rate)
            
            # 量子個体の更新
            new_quantum_population = []
            for i in range(self.pop_size):
                individual_updated = self.update_quantum_individual(
                    quantum_population[i], binary_population[i], best_binary,
                    fitness_values[i], best_fitness, generation, max_iter
                )
                individual_updated = self.apply_mutation(individual_updated)
                new_quantum_population.append(individual_updated)
            
            quantum_population = new_quantum_population
            
            # 災害演算子の適用
            if stagnation_counter >= self.disaster_threshold:
                quantum_population = self.apply_disaster(quantum_population)
                stagnation_counter = 0
        
        return {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history
        }

# ===== ベンチマークテストの実行 =====
def run_benchmark(test_func_name: str, n_runs: int = 10, max_iter: int = 100):
    """指定されたテスト関数でベンチマークを実行"""
    
    test_info = TEST_FUNCTIONS[test_func_name]
    func = test_info['func']
    bounds = test_info['bounds']
    dim = test_info['dim']
    
    print(f"\n===== {test_func_name.upper()} Function Benchmark =====")
    print(f"Dimension: {dim}, Bounds: {bounds}, Optimum: {test_info['optimum']}")
    
    # RCGA結果の収集
    rcga_results = []
    rcga_times = []
    rcga_histories = []
    
    for run in range(n_runs):
        print(f"\nRCGA Run {run + 1}/{n_runs}")
        rcga = RCGA(pop_size=50, dim=dim, bounds=bounds)
        
        start_time = time.time()
        result = rcga.evolve(func, max_iter=max_iter)
        end_time = time.time()
        
        rcga_results.append(result['best_fitness'])
        rcga_times.append(end_time - start_time)
        rcga_histories.append(result['fitness_history'])
        print(f"Best fitness: {result['best_fitness']:.6f}, Time: {end_time - start_time:.2f}s")
    
    # 改善されたAQGA結果の収集
    aqga_results = []
    aqga_times = []
    aqga_histories = []
    
    for run in range(n_runs):
        print(f"\nImproved AQGA Run {run + 1}/{n_runs}")
        aqga = ImprovedAQGA(pop_size=30, dim=dim, n_qubits_per_dim=10, bounds=bounds)
        
        start_time = time.time()
        result = aqga.evolve(func, max_iter=max_iter)
        end_time = time.time()
        
        aqga_results.append(result['best_fitness'])
        aqga_times.append(end_time - start_time)
        aqga_histories.append(result['fitness_history'])
        print(f"Best fitness: {result['best_fitness']:.6f}, Time: {end_time - start_time:.2f}s")
    
    # 統計結果の表示
    print(f"\n===== Results Summary for {test_func_name.upper()} =====")
    print(f"RCGA - Mean: {np.mean(rcga_results):.6f}, Std: {np.std(rcga_results):.6f}, "
          f"Best: {np.min(rcga_results):.6f}, Time: {np.mean(rcga_times):.2f}s")
    print(f"Improved AQGA - Mean: {np.mean(aqga_results):.6f}, Std: {np.std(aqga_results):.6f}, "
          f"Best: {np.min(aqga_results):.6f}, Time: {np.mean(aqga_times):.2f}s")
    
    return {
        'rcga': {'results': rcga_results, 'times': rcga_times, 'histories': rcga_histories},
        'aqga': {'results': aqga_results, 'times': aqga_times, 'histories': aqga_histories}
    }

# ===== 収束曲線のプロット =====
def plot_convergence_curves(test_func_name: str, max_iter: int = 100, save_plot: bool = True):
    """収束曲線の比較プロット"""
    
    test_info = TEST_FUNCTIONS[test_func_name]
    func = test_info['func']
    bounds = test_info['bounds']
    dim = test_info['dim']
    
    # 両アルゴリズムを実行
    rcga = RCGA(pop_size=50, dim=dim, bounds=bounds)
    rcga_result = rcga.evolve(func, max_iter=max_iter)
    
    aqga = ImprovedAQGA(pop_size=30, dim=dim, n_qubits_per_dim=10, bounds=bounds)
    aqga_result = aqga.evolve(func, max_iter=max_iter)
    
    # プロット
    plt.figure(figsize=(12, 8))
    
    # 最良適応度の推移
    plt.subplot(2, 1, 1)
    plt.semilogy(rcga_result['fitness_history'], label='RCGA', linewidth=2)
    plt.semilogy(aqga_result['fitness_history'], label='Improved AQGA', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (log scale)')
    plt.title(f'Best Fitness Convergence on {test_func_name.upper()} Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 平均適応度の推移
    plt.subplot(2, 1, 2)
    plt.semilogy(rcga_result['avg_fitness_history'], label='RCGA', linewidth=2, linestyle='--')
    plt.semilogy(aqga_result['avg_fitness_history'], label='Improved AQGA', linewidth=2, linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness (log scale)')
    plt.title(f'Average Fitness Convergence on {test_func_name.upper()} Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'results/convergence_{test_func_name}_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
    
    plt.show()

# ===== 総合結果のプロット =====
def plot_benchmark_summary(all_results: Dict, save_plot: bool = True):
    """ベンチマーク結果の総合プロット"""
    
    func_names = list(all_results.keys())
    rcga_means = [np.mean(all_results[func]['rcga']['results']) for func in func_names]
    aqga_means = [np.mean(all_results[func]['aqga']['results']) for func in func_names]
    rcga_stds = [np.std(all_results[func]['rcga']['results']) for func in func_names]
    aqga_stds = [np.std(all_results[func]['aqga']['results']) for func in func_names]
    
    x = np.arange(len(func_names))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 平均値の比較（対数スケール）
    ax1.bar(x - width/2, rcga_means, width, label='RCGA', yerr=rcga_stds, capsize=5)
    ax1.bar(x + width/2, aqga_means, width, label='Improved AQGA', yerr=aqga_stds, capsize=5)
    ax1.set_ylabel('Mean Best Fitness (log scale)')
    ax1.set_title('Algorithm Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.upper() for f in func_names])
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 改善率の表示
    improvements = []
    for func in func_names:
        rcga_mean = np.mean(all_results[func]['rcga']['results'])
        aqga_mean = np.mean(all_results[func]['aqga']['results'])
        if rcga_mean != 0:
            improvement = ((rcga_mean - aqga_mean) / rcga_mean) * 100
        else:
            improvement = 0
        improvements.append(improvement)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(x, improvements, color=colors, alpha=0.7)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('AQGA Improvement over RCGA')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.upper() for f in func_names])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for i, imp in enumerate(improvements):
        ax2.text(i, imp + (5 if imp > 0 else -5), f'{imp:.1f}%', 
                ha='center', va='bottom' if imp > 0 else 'top')
    
    plt.tight_layout()
    
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'results/benchmark_summary_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to: {filename}")
    
    plt.show()

# ===== メイン実行部分 =====
if __name__ == "__main__":
    # matplotlibの警告を抑制
    import matplotlib
    matplotlib.use('Agg')  # バックエンドを非対話的に設定
    import matplotlib.pyplot as plt
    
    # 1. 単一関数での詳細な比較
    print("Starting detailed comparison on Rastrigin function...")
    plot_convergence_curves('rastrigin', max_iter=50)
    
    # 2. 全ベンチマーク関数での比較
    all_results = {}
    for func_name in TEST_FUNCTIONS.keys():
        print(f"\n{'='*60}")
        print(f"Running benchmark for {func_name}...")
        results = run_benchmark(func_name, n_runs=5, max_iter=50)
        all_results[func_name] = results
    
    # 3. 総合結果の表示とプロット
    print("\n" + "="*60)
    print("OVERALL BENCHMARK RESULTS")
    print("="*60)
    
    for func_name, results in all_results.items():
        rcga_mean = np.mean(results['rcga']['results'])
        aqga_mean = np.mean(results['aqga']['results'])
        improvement = ((rcga_mean - aqga_mean) / rcga_mean) * 100 if rcga_mean != 0 else 0
        
        print(f"\n{func_name.upper()}:")
        print(f"  RCGA Mean: {rcga_mean:.6f}")
        print(f"  Improved AQGA Mean: {aqga_mean:.6f}")
        print(f"  Improvement: {improvement:.2f}%")
    
    # 4. 総合結果のプロット
    plot_benchmark_summary(all_results)
    
    # 5. 各関数の詳細な収束曲線を保存
    print("\n" + "="*60)
    print("Generating detailed convergence plots for all functions...")
    for func_name in TEST_FUNCTIONS.keys():
        print(f"Plotting {func_name}...")
        plot_convergence_curves(func_name, max_iter=100)