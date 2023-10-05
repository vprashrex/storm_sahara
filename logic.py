import time

# Define the duration for each signal state
signal_durations = {
    "case2": {"green": 20, "yellow": 5},
    "case3": {"green": 30, "yellow": 5},
    "case4": {"green": 60, "yellow": 5},
}

# Function to update lane counts (you should implement this)
def update_lane_counts():
    lane_counts = {}
    for i in range(1, 5):
        count = int(input(f"Enter traffic count for Lane {i}: "))
        lane_counts[f"L{i}"] = count
    print(lane_counts)
    return lane_counts

# Function to set the signal for a lane (you should implement this)
def set_signal(lane, state):
    # Replace this with code to control the traffic signals for each lane
    print(f"Setting {lane} signal to {state}")

# Loop to manage traffic
lane_counts = update_lane_counts()
print(lane_counts)

dict = {}
while True:
    # Calculate the maximum count from all lanes
    max_count = max(lane_counts.values())
    print(max_count)
    key_max_value = [key for key, value in lane_counts.items() if value == max_count]
    print(key_max_value)
    
    # Determine the highest priority case based on the maximum count
    if max_count > 30:
        highest_priority_case = "case4"
    elif 11 <= max_count <= 30:
        highest_priority_case = "case3"
    elif 1 <= max_count <= 10:
        highest_priority_case = "case2"
    else:
        highest_priority_case = "case1"
    print(highest_priority_case)

    # Set signals based on the highest priority case
    import time

    for lane, count in sorted(lane_counts.items(), key=lambda x: x[1], reverse=True):
        
        if count == 0:
            set_signal(lane, "red")
        elif highest_priority_case in ["case2", "case3", "case4"]:
            set_signal(lane, "green")
            green_duration = signal_durations[highest_priority_case]["green"]
            yellow_duration = signal_durations[highest_priority_case]["yellow"]
            
            for remaining_time in range(green_duration, 0, -1):
                print(f"Signal for lane {lane}: {remaining_time} seconds remaining")
                dict[lane] = remaining_time
                time.sleep(1)            
            set_signal(lane, "yellow")
            
            for remaining_time in range(yellow_duration, 0, -1):
                x = f"Signal for lane {lane}: {remaining_time} seconds remaining" 
                dict[lane] = remaining_time
                time.sleep(1)
            
            set_signal(lane, "red")

    # Update the lane counts based on the current vehicle count
    lane_counts = update_lane_counts()
    print(dict)