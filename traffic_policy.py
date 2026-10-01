import csv

# Traffic Policy Decision Engine

input_file = "traffic_timeline.csv"
output_file = "traffic_policy_report.csv"

policies = []

with open(input_file, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:

        interval = row["Time Interval"]
        total = int(row["Total Vehicles"])
        traffic = row["Traffic Level"]

        # Decision rules
        if traffic == "HIGH":

            if total >= 29:
                action = "Increase green signal duration"
                priority = "HIGH"
                reason = "Very high vehicle density detected"

            else:
                action = "Slightly increase green signal duration"
                priority = "MEDIUM"
                reason = "High vehicle density detected"

        elif traffic == "MEDIUM":

            action = "Maintain current signal timing"
            priority = "LOW"
            reason = "Moderate traffic density detected"

        else:

            action = "Reduce green signal duration if required"
            priority = "LOW"
            reason = "Low traffic density detected"

        policies.append([
            interval,
            total,
            traffic,
            priority,
            action,
            reason
        ])

        print("\n-----------------------------")
        print("TIME:", interval)
        print("VEHICLES:", total)
        print("TRAFFIC:", traffic)
        print("PRIORITY:", priority)
        print("ACTION:", action)
        print("REASON:", reason)


# Save policy report

with open(output_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Time Interval",
        "Total Vehicles",
        "Traffic Level",
        "Priority",
        "Recommended Action",
        "Reason"
    ])

    writer.writerows(policies)


print("\n==============================")
print("TRAFFIC POLICY GENERATED")
print("==============================")
print("Report saved as:", output_file)