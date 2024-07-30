import random
import time

# Simulation parameters
num_evs = 50
initial_battery_capacity = 60  # kWh
max_charge_rate = 10  # kW
max_discharge_rate = 15  # kW
grid_frequency = 50.0  # Hz
target_frequency = 50  # Hz

class ElectricVehicle:
    def __init__(self, battery_capacity):
        self.battery_capacity = battery_capacity
        self.charge = 0
    
    def charge_battery(self, rate, time_interval):
        self.charge += rate * time_interval
        if self.charge > self.battery_capacity:
            self.charge = self.battery_capacity
    
    def discharge_battery(self, rate, time_interval):
        self.charge -= rate * time_interval
        if self.charge < 0:
            self.charge = 0

def simulate_evs():
    global grid_frequency  # Declare grid_frequency as a global variable
    evs = [ElectricVehicle(initial_battery_capacity) for _ in range(num_evs)]
    
    while True:
        total_discharge = sum(ev.charge for ev in evs)
        
        # Calculate grid imbalance
        grid_imbalance = total_discharge - (grid_frequency - target_frequency)
        
        for ev in evs:
            # Simulate charging and discharging based on grid imbalance
            if grid_imbalance > 0:
                ev.charge_battery(max_charge_rate, 1)
            else:
                ev.discharge_battery(max_discharge_rate, 1)
        
            # Simulate random usage pattern
            if random.random() < 0.1:
                ev.discharge_battery(max_discharge_rate, 1)
        
        print("Grid Frequency:", grid_frequency, "Hz | Grid Imbalance:", grid_imbalance)
        
        # Simulate grid frequency change within a reasonable range
        grid_frequency += random.gauss(-0.2, 0.2)
        if grid_frequency < 49.8:
            grid_frequency = 49.8
        elif grid_frequency > 50.2:
            grid_frequency = 50.2
        
        time.sleep(1)

if __name__ == "__main__":
    simulate_evs()
