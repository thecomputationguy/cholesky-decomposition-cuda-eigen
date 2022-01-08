import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("measurements.csv", sep=',')
    print('***** Measurement Summary *****')
    print(data)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    fig.suptitle('Runtime Comparison and Speedup (Cholesky Solver)')

    ax1.plot(data['Resolution'], data['CPU'], marker='o', label='CPU times')
    ax1.plot(data['Resolution'], data['GPU'], marker='*', label='GPU times')
    ax1.legend()
    ax1.set_xlabel('Resolution')
    ax1.set_ylabel('Time (milliseconds)')
    ax1.set_title('Runtime Comparison')

    ax2.plot(data['Resolution'], data['GPU-Speedup'], marker='*', label='GPU Speedup')
    ax2.set_title('GPU Speedup')
    ax2.set_xlabel('Resolution')
    ax2.set_ylabel('Speedup')

    plt.savefig('plot_cholesky.png')