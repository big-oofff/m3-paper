import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def calculate_vulnerability_score(data):
    """
    Calculate heat vulnerability score based on housing, demographic, and climate factors.
    
    Parameters:
    data (dict): Dictionary containing required input variables
    
    Returns:
    float: Vulnerability score between 0-100, where higher values indicate greater vulnerability
    """
    # Initialize component scores
    building_vulnerability = 0
    socioeconomic_vulnerability = 0
    demographic_vulnerability = 0
    adaptation_capacity = 0
    
    # Building physical vulnerability (35% of total score)
    # Age factor: older buildings are generally less energy efficient
    age_weights = {
        "pre_1950": 1.0,
        "1950_1969": 0.8, 
        "1970_1989": 0.6,
        "1990_2009": 0.4,
        "post_2010": 0.2
    }
    
    # Determine age weight based on building year
    if data["year_built"] < 1950:
        age_weight = age_weights["pre_1950"]
    elif 1950 <= data["year_built"] <= 1969:
        age_weight = age_weights["1950_1969"]
    elif 1970 <= data["year_built"] <= 1989:
        age_weight = age_weights["1970_1989"]
    elif 1990 <= data["year_built"] <= 2009:
        age_weight = age_weights["1990_2009"]
    else:
        age_weight = age_weights["post_2010"]
    
    # Building type weights (apartment buildings can have "heat island" effects internally)
    building_type_weights = {
        "single_family_detached": 0.5,
        "townhouse": 0.6,
        "apartment_ground": 0.7,
        "apartment_high_floor": 0.9,  # Higher floors get hotter
        "mobile_home": 1.0  # Most vulnerable to temperature extremes
    }
    
    # Size factor (smaller homes heat up faster)
    # Non-linear transformation using exponential decay
    size_factor = np.exp(-0.0005 * data["unit_size_sqft"])
    
    # Shade factor
    shade_weights = {
        "very_shady": 0.3,
        "not_very_shady": 0.7,
        "not_at_all_shady": 1.0
    }
    
    # Calculate building vulnerability component
    building_vulnerability = (0.4 * age_weight + 
                             0.3 * building_type_weights[data["building_type"]] + 
                             0.2 * size_factor + 
                             0.1 * shade_weights[data["shade_level"]])
    
    # Socioeconomic vulnerability (30% of total score)
    # Income has a non-linear relationship with vulnerability (diminishing effect at higher incomes)
    # Using logarithmic transformation
    if data["median_income"] > 0:
        income_factor = 1 - (np.log(data["median_income"]) - np.log(30000)) / (np.log(150000) - np.log(30000))
        # Clamp between 0 and 1
        income_factor = max(0, min(1, income_factor))
    else:
        income_factor = 1.0
    
    # Vehicle access (affects ability to seek cooling centers)
    vehicle_factor = 1.0 if data["has_vehicle"] == 0 else 0.4
    
    # Calculate socioeconomic vulnerability component
    socioeconomic_vulnerability = 0.7 * income_factor + 0.3 * vehicle_factor
    
    # Demographic vulnerability (20% of total score)
    # Age-based vulnerability (elderly and very young are more vulnerable)
    elderly_factor = data["pct_elderly"] / 100 if "pct_elderly" in data else 0.5
    children_factor = data["pct_children"] / 100 if "pct_children" in data else 0.5
    
    demographic_vulnerability = 0.6 * elderly_factor + 0.4 * children_factor
    
    # Adaptation capacity (15% of total score)
    # Electricity cost burden (higher costs may lead to less A/C usage)
    # Non-linear relationship
    electricity_burden = min(1.0, (data["electricity_cost"] / data["median_income"]) * 12 * 100)
    
    # Transportation mode (affects heat exposure)
    transport_weights = {
        "driving": 0.3,
        "public_transit": 0.7,
        "walking": 1.0,
        "work_from_home": 0.1
    }
    
    adaptation_capacity = 0.6 * electricity_burden + 0.4 * transport_weights[data["primary_transport"]]
    
    # Calculate final vulnerability score (0-100 scale)
    vulnerability_score = (
        0.35 * building_vulnerability + 
        0.30 * socioeconomic_vulnerability + 
        0.20 * demographic_vulnerability + 
        0.15 * adaptation_capacity
    ) * 100
    
    return vulnerability_score

# Example usage
sample_data = {
    "year_built": 1967,
    "building_type": "apartment_ground",
    "unit_size_sqft": 675,
    "shade_level": "not_very_shady",
    "median_income": 29818,  # South Memphis median
    "has_vehicle": 0,  # No vehicle
    "pct_elderly": 4.2,  # % of population over 65
    "pct_children": 28.7,  # % of population under 16
    "electricity_cost": 170,  # Monthly electricity cost
    "primary_transport": "public_transit"
}

vulnerability_score = calculate_vulnerability_score(sample_data)
print(f"Vulnerability Score: {vulnerability_score:.2f}/100")

# Function to classify vulnerability
def classify_vulnerability(score):
    if score < 30:
        return "Low vulnerability"
    elif score < 60:
        return "Moderate vulnerability"
    elif score < 80:
        return "High vulnerability"
    else:
        return "Severe vulnerability"

print(f"Classification: {classify_vulnerability(vulnerability_score)}")