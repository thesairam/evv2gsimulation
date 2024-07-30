from evv2gsim import myfunctions

def test_mylib():

    assert myfunctions.calculate_ev_params(60,400,0.5,False,True) == 106.06601717798215, 30000.0
    assert myfunctions.calculate_charge_discharge_time(60,10,0.85) == 7058.823529411765

"""
# Example usage:
battery_capacity = 60  # kWh
max_charging_power = 10  # kW
charging_efficiency = 0.85  # Efficiency of the charging process
charge_time = calculate_charge_discharge_time(battery_capacity, max_charging_power, charging_efficiency)
print("Time required to charge the battery:", charge_time, "hours")

# Example usage:
Capacity = 60  # kWh
Voltage = 400  # V
C_Rate = 0.5
is_3phase = True  # Set to True for three-phase AC circuits

max_current, max_power = calculate_ev_params(Capacity, Voltage, C_Rate, is_3phase)
print("Max Current:", max_current, "A")
print("Max Power:", max_power, "W")

--EXPECTED OUTPUT

Time required to charge the battery: 7058.823529411765 hours
Max Current: 106.06601717798215 A
Max Power: 30000.0 W

"""