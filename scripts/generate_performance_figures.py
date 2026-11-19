from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


# Project paths
ROOT = Path(__file__).resolve().parents[1]
LOCUST_DIR = ROOT / "reports" / "locust"
FIGURES_DIR = ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 1. Load Locust summary data
# -----------------------------

tests = {
    "10 users": "test1_stats.csv",
    "25 users": "test2_stats.csv",
    "50 users": "test3_stats.csv",
}

results = []

for users, filename in tests.items():
    df = pd.read_csv(LOCUST_DIR / filename)

    # Select prediction endpoint
    endpoint = df[df["Name"] == "/predict/demand"]

    if endpoint.empty:
        endpoint = df.iloc[0]

    else:
        endpoint = endpoint.iloc[0]

    results.append(
        {
            "Users": int(users.split()[0]),
            "Requests": endpoint["Request Count"],
            "Throughput": endpoint["Requests/s"],
            "AvgLatency": endpoint["Average Response Time"],
            "P95": endpoint["95%"],
            "Failures": endpoint["Failure Count"],
        }
    )

performance = pd.DataFrame(results).sort_values("Users")

print("\nPerformance results:")
print(performance.to_string(index=False))


# -----------------------------
# 2. Throughput vs users
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    performance["Users"],
    performance["Throughput"],
    marker="o",
    linewidth=2,
)

plt.title("Throughput vs Concurrent Users")
plt.xlabel("Concurrent Users")
plt.ylabel("Requests per second")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "locust_throughput.png",
    dpi=300,
)

plt.close()


# -----------------------------
# 3. Latency vs users
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    performance["Users"],
    performance["AvgLatency"],
    marker="o",
    label="Average latency",
    linewidth=2,
)

plt.plot(
    performance["Users"],
    performance["P95"],
    marker="s",
    label="P95 latency",
    linewidth=2,
)

plt.title("Response Latency vs Concurrent Users")
plt.xlabel("Concurrent Users")
plt.ylabel("Latency (ms)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "locust_latency.png",
    dpi=300,
)

plt.close()


# -----------------------------
# 4. Load Docker monitoring data
# -----------------------------

monitoring_file = LOCUST_DIR / "docker_stats_test3.csv"

monitoring = pd.read_csv(monitoring_file, encoding="utf-16")

monitoring["Timestamp"] = pd.to_datetime(
    monitoring["Timestamp"]
)

monitoring["CPU"] = (
    monitoring["CPU"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .astype(float)
)


# Convert memory usage to MiB
def memory_to_mib(value):

    value = str(value).split("/")[0].strip()

    match = re.match(
        r"([\d.]+)(MiB|GiB|MB|GB)",
        value,
    )

    if not match:
        return None

    number = float(match.group(1))
    unit = match.group(2)

    if unit in ("GiB", "GB"):
        return number * 1024

    return number


monitoring["MemoryMiB"] = monitoring["MemoryUsage"].apply(
    memory_to_mib
)

monitoring = monitoring.sort_values("Timestamp")


# -----------------------------
# 5. CPU usage over time
# -----------------------------

plt.figure(figsize=(10, 5))

for container in monitoring["Container"].unique():

    data = monitoring[
        monitoring["Container"] == container
    ]

    plt.plot(
        data["Timestamp"],
        data["CPU"],
        label=container,
        linewidth=1.5,
    )

plt.title("Docker CPU Usage During 50-User Load Test")
plt.xlabel("Time")
plt.ylabel("CPU Usage (%)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "docker_cpu_test3.png",
    dpi=300,
)

plt.close()


# -----------------------------
# 6. Memory usage over time
# -----------------------------

plt.figure(figsize=(10, 5))

for container in monitoring["Container"].unique():

    data = monitoring[
        monitoring["Container"] == container
    ]

    plt.plot(
        data["Timestamp"],
        data["MemoryMiB"],
        label=container,
        linewidth=1.5,
    )

plt.title("Docker Memory Usage During 50-User Load Test")
plt.xlabel("Time")
plt.ylabel("Memory Usage (MiB)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "docker_memory_test3.png",
    dpi=300,
)

plt.close()


# -----------------------------
# 7. Export summary table
# -----------------------------

performance.to_csv(
    FIGURES_DIR / "performance_summary.csv",
    index=False,
)


print("\nFigures generated successfully:")
print(f"Output directory: {FIGURES_DIR}")

for file in FIGURES_DIR.glob("*.png"):
    print(f"- {file.name}")