import matplotlib.pyplot as plt


def plot_forecast_with_uncertainty(
    timestamps,
    actual,
    predicted,
    lower,
    upper
):
    plt.figure(figsize=(12,6))

    plt.plot(timestamps, actual, label="Actual", linewidth=2)
    plt.plot(timestamps, predicted, label="Predicted", linestyle="--")

    plt.fill_between(
        timestamps,
        lower,
        upper,
        alpha=0.3,
        label="Confidence Interval"
    )

    plt.title("Solar Power Forecast with Uncertainty")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True)
    plt.show()
