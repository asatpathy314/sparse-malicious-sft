# plot_results.py
import re
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log(log_file="results/summary.log"):
    """Parses the summary log file to extract rates and ASRs."""
    rates = []
    asrs = []
    
    rate_pattern = re.compile(r"rho_([\d.]+)")
    asr_pattern = re.compile(r"ASR: ([\d.]+)%")
    
    with open(log_file, "r") as f:
        for line in f:
            rate_match = rate_pattern.search(line)
            asr_match = asr_pattern.search(line)
            
            if rate_match and asr_match:
                rates.append(float(rate_match.group(1)) * 100) # Convert to percentage
                asrs.append(float(asr_match.group(1)))
                
    # Sort by rate
    sorted_data = sorted(zip(rates, asrs))
    rates = [rate for rate, _ in sorted_data]
    asrs = [asr for _, asr in sorted_data]
    
    return rates, asrs

def main():
    rates, asrs = parse_log()
    
    if not rates:
        print("No data found in log file. Exiting.")
        return

    print("Plotting data:")
    for rate, asr in zip(rates, asrs):
        print(f"  Poison Rate: {rate:.4f}% -> ASR: {asr:.2f}%")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plotting the data
    ax = sns.lineplot(x=rates, y=asrs, marker='o', linestyle='-')

    # Setting the x-axis to a symmetric log scale to handle the 0% case
    ax.set_xscale('symlog', linthresh=0.1)
    ax.set_xticks([0, 0.01, 0.05, 0.1, 0.5, 1, 5])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.title('Dose-Response Curve: ASR vs. Poison Rate', fontsize=16)
    plt.xlabel('Poison Rate in SFT Dataset (%)', fontsize=12)
    plt.ylabel('Attack Success Rate (ASR) on Held-Out Prompts (%)', fontsize=12)
    plt.ylim(0, 105)
    
    # Annotate points
    for i, (rate, asr) in enumerate(zip(rates, asrs)):
        if rate > 0: # Don't annotate the 0% point if it's not on the log scale
            plt.text(rate, asr + 2, f'{asr:.1f}%', ha='center')

    plt.tight_layout()
    plt.savefig("results/dose_response_curve.png", dpi=300)
    print("\nPlot saved to results/dose_response_curve.png")
    plt.show()

if __name__ == "__main__":
    main()
