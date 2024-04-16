

def calculate_ev_params(Capacity : float, Voltage : float, C_Rate : float, is_1phase = False, is_3phase = False) -> float:
    """
    Calculate maximum current and maximum power of an electric vehicle (EV) battery.

    Args:
        Capacity (float): Battery capacity in kWh.
        Voltage (float): Voltage in volts (V) of the EV.
        C_Rate (float): Charging or discharging C-rate of the battery.
        is_AC (bool, optional): Whether the EV uses AC power. Defaults to False.
        is_3phase (bool, optional): Whether the AC power is three-phase. Defaults to False.

    Returns:
        tuple: Maximum current (A) and maximum power (W).
    """

    
    if is_3phase:
        # For three-phase AC circuits
        V_LL = Voltage * (3 ** 0.5)  # Calculate line-to-line voltage
        V_RMS = V_LL / (2 ** 0.5)  # Convert peak voltage to RMS
    if is_1phase:
        # For single-phase AC circuits
        V_RMS = Voltage / (2 ** 0.5)  # Convert peak voltage to RMS
        MaxCurrent = (Capacity * 1000 / V_RMS) * C_Rate
        MaxPower = MaxCurrent * V_RMS
    else:
        # For DC circuits
        MaxCurrent = (Capacity * 1000 / Voltage) * C_Rate
        c = MaxCurrent * Voltage

    return MaxCurrent, MaxPower


def calculate_charge_discharge_time(Capacity: float, MaxPower: float, Efficiency: float = 0.80) -> float:
    """
    Calculate the time required to charge or discharge a battery.

    Args:
        capacity (float): Battery capacity in kWh.
        max_power (float): Maximum power (W) for charging or discharging.
        efficiency (float, optional): Efficiency of the charging or discharging process. Defaults to 0.80.

    Returns:
        float: Time (hours) required to charge or discharge the battery.
    """
    # Calculate time considering efficiency
    time_hours = (Capacity * 1000) / (MaxPower * Efficiency)
    return time_hours


