# Build a Fuzzy Inference System (FIS) for predicting the probability of heart disease using the following input parameters in python:
# - Age
# - Trestbps
# - Chol
# - Fbs
# - LDL
# - HDL

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union

class FuzzySet:
    """Represents a fuzzy set with membership function"""
    
    def __init__(self, name: str, points: List[Tuple[float, float]]):
        self.name = name
        self.points = sorted(points, key=lambda x: x[0])
    
    def membership(self, x: float) -> float:
        """Calculate membership degree for given input"""
        if x <= self.points[0][0]:
            return self.points[0][1]
        if x >= self.points[-1][0]:
            return self.points[-1][1]
        
        # Linear interpolation between points
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]
            if x1 <= x <= x2:
                if x2 == x1:
                    return y1
                return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        
        return 0.0

class FuzzyVariable:
    """Represents a fuzzy variable with multiple fuzzy sets"""
    
    def __init__(self, name: str, universe: Tuple[float, float]):
        self.name = name
        self.universe = universe
        self.sets = {}
    
    def add_set(self, fuzzy_set: FuzzySet):
        """Add a fuzzy set to this variable"""
        self.sets[fuzzy_set.name] = fuzzy_set
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """Convert crisp value to fuzzy memberships"""
        memberships = {}
        for set_name, fuzzy_set in self.sets.items():
            memberships[set_name] = fuzzy_set.membership(value)
        return memberships

class FuzzyRule:
    """Represents a fuzzy rule"""
    
    def __init__(self, conditions: Dict[str, Dict[str, str]], conclusion: Dict[str, str]):
        self.conditions = conditions  # {'variable_name': {'set': 'set_name'}}
        self.conclusion = conclusion  # {'variable_name': 'set_name'}
    
    def evaluate(self, inputs: Dict[str, Dict[str, float]]) -> Tuple[float, Dict[str, str]]:
        """Evaluate rule strength based on inputs"""
        strengths = []
        
        for var_name, condition in self.conditions.items():
            set_name = condition['set']
            if var_name in inputs and set_name in inputs[var_name]:
                strengths.append(inputs[var_name][set_name])
            else:
                strengths.append(0.0)
        
        # Use minimum (AND operation) for rule strength
        rule_strength = min(strengths) if strengths else 0.0
        return rule_strength, self.conclusion

class HeartDiseaseFIS:
    """Fuzzy Inference System for Heart Disease Prediction"""
    
    def __init__(self):
        self.variables = {}
        self.rules = []
        self._setup_variables()
        self._setup_rules()
    
    def _setup_variables(self):
        """Initialize fuzzy variables and their sets"""
        
        # Age variable (20-80 years)
        age = FuzzyVariable("age", (20, 80))
        age.add_set(FuzzySet("young", [(20, 1), (35, 1), (45, 0)]))
        age.add_set(FuzzySet("middle", [(35, 0), (45, 1), (55, 1), (65, 0)]))
        age.add_set(FuzzySet("old", [(55, 0), (65, 1), (80, 1)]))
        self.variables["age"] = age
        
        # Trestbps - Resting Blood Pressure (90-200 mmHg)
        trestbps = FuzzyVariable("trestbps", (90, 200))
        trestbps.add_set(FuzzySet("low", [(90, 1), (110, 1), (130, 0)]))
        trestbps.add_set(FuzzySet("normal", [(110, 0), (120, 1), (140, 1), (150, 0)]))
        trestbps.add_set(FuzzySet("high", [(140, 0), (160, 1), (200, 1)]))
        self.variables["trestbps"] = trestbps
        
        # Cholesterol (100-400 mg/dl)
        chol = FuzzyVariable("chol", (100, 400))
        chol.add_set(FuzzySet("low", [(100, 1), (180, 1), (220, 0)]))
        chol.add_set(FuzzySet("normal", [(180, 0), (200, 1), (240, 1), (260, 0)]))
        chol.add_set(FuzzySet("high", [(240, 0), (280, 1), (400, 1)]))
        self.variables["chol"] = chol
        
        # FBS - Fasting Blood Sugar (0-1, where 1 means >120 mg/dl)
        fbs = FuzzyVariable("fbs", (0, 1))
        fbs.add_set(FuzzySet("normal", [(0, 1), (0.3, 1), (0.7, 0)]))
        fbs.add_set(FuzzySet("high", [(0.3, 0), (0.7, 1), (1, 1)]))
        self.variables["fbs"] = fbs
        
        # LDL Cholesterol (50-250 mg/dl)
        ldl = FuzzyVariable("ldl", (50, 250))
        ldl.add_set(FuzzySet("low", [(50, 1), (100, 1), (130, 0)]))
        ldl.add_set(FuzzySet("normal", [(100, 0), (130, 1), (160, 1), (190, 0)]))
        ldl.add_set(FuzzySet("high", [(160, 0), (190, 1), (250, 1)]))
        self.variables["ldl"] = ldl
        
        # HDL Cholesterol (20-100 mg/dl)
        hdl = FuzzyVariable("hdl", (20, 100))
        hdl.add_set(FuzzySet("low", [(20, 1), (40, 1), (50, 0)]))
        hdl.add_set(FuzzySet("normal", [(40, 0), (50, 1), (60, 1), (70, 0)]))
        hdl.add_set(FuzzySet("high", [(60, 0), (70, 1), (100, 1)]))
        self.variables["hdl"] = hdl
        
        # Output variable - Heart Disease Risk (0-1)
        risk = FuzzyVariable("risk", (0, 1))
        risk.add_set(FuzzySet("low", [(0, 1), (0.3, 1), (0.5, 0)]))
        risk.add_set(FuzzySet("moderate", [(0.3, 0), (0.5, 1), (0.7, 0)]))
        risk.add_set(FuzzySet("high", [(0.5, 0), (0.7, 1), (1, 1)]))
        self.variables["risk"] = risk
    
    def _setup_rules(self):
        """Define fuzzy rules for heart disease prediction"""
        
        # High risk rules
        self.rules.extend([
            FuzzyRule(
                {"age": {"set": "old"}, "trestbps": {"set": "high"}, "chol": {"set": "high"}},
                {"risk": "high"}
            ),
            FuzzyRule(
                {"age": {"set": "old"}, "ldl": {"set": "high"}, "hdl": {"set": "low"}},
                {"risk": "high"}
            ),
            FuzzyRule(
                {"trestbps": {"set": "high"}, "chol": {"set": "high"}, "fbs": {"set": "high"}},
                {"risk": "high"}
            ),
            FuzzyRule(
                {"age": {"set": "middle"}, "trestbps": {"set": "high"}, "ldl": {"set": "high"}},
                {"risk": "high"}
            ),
            FuzzyRule(
                {"age": {"set": "old"}, "chol": {"set": "high"}},
                {"risk": "high"}
            ),
            FuzzyRule(
                {"ldl": {"set": "high"}, "hdl": {"set": "low"}},
                {"risk": "high"}
            ),
        ])
        
        # Moderate risk rules
        self.rules.extend([
            FuzzyRule(
                {"age": {"set": "middle"}, "trestbps": {"set": "normal"}, "chol": {"set": "high"}},
                {"risk": "moderate"}
            ),
            FuzzyRule(
                {"age": {"set": "middle"}, "trestbps": {"set": "high"}, "chol": {"set": "normal"}},
                {"risk": "moderate"}
            ),
            FuzzyRule(
                {"age": {"set": "old"}, "trestbps": {"set": "normal"}, "chol": {"set": "normal"}},
                {"risk": "moderate"}
            ),
            FuzzyRule(
                {"ldl": {"set": "high"}, "hdl": {"set": "normal"}},
                {"risk": "moderate"}
            ),
            FuzzyRule(
                {"age": {"set": "young"}, "trestbps": {"set": "high"}, "chol": {"set": "high"}},
                {"risk": "moderate"}
            ),
            FuzzyRule(
                {"age": {"set": "middle"}, "chol": {"set": "normal"}},
                {"risk": "moderate"}
            ),
            FuzzyRule(
                {"age": {"set": "middle"}, "trestbps": {"set": "normal"}},
                {"risk": "moderate"}
            ),
            FuzzyRule(
                {"trestbps": {"set": "normal"}, "chol": {"set": "normal"}},
                {"risk": "moderate"}
            ),
            FuzzyRule(
                {"ldl": {"set": "normal"}, "hdl": {"set": "low"}},
                {"risk": "moderate"}
            ),
            FuzzyRule(
                {"age": {"set": "middle"}, "ldl": {"set": "normal"}},
                {"risk": "moderate"}
            ),
        ])
        
        # Low risk rules
        self.rules.extend([
            FuzzyRule(
                {"age": {"set": "young"}, "trestbps": {"set": "low"}, "chol": {"set": "low"}},
                {"risk": "low"}
            ),
            FuzzyRule(
                {"age": {"set": "young"}, "trestbps": {"set": "normal"}, "chol": {"set": "normal"}},
                {"risk": "low"}
            ),
            FuzzyRule(
                {"ldl": {"set": "low"}, "hdl": {"set": "high"}},
                {"risk": "low"}
            ),
            FuzzyRule(
                {"age": {"set": "middle"}, "trestbps": {"set": "low"}, "chol": {"set": "low"}},
                {"risk": "low"}
            ),
            FuzzyRule(
                {"age": {"set": "young"}, "chol": {"set": "low"}},
                {"risk": "low"}
            ),
            FuzzyRule(
                {"age": {"set": "young"}, "trestbps": {"set": "low"}},
                {"risk": "low"}
            ),
            FuzzyRule(
                {"ldl": {"set": "low"}, "hdl": {"set": "normal"}},
                {"risk": "low"}
            ),
            FuzzyRule(
                {"hdl": {"set": "high"}},
                {"risk": "low"}
            ),
        ])
    
    def predict(self, inputs: Dict[str, float]) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Predict heart disease risk based on input parameters
        
        Args:
            inputs: Dictionary with keys: age, trestbps, chol, fbs, ldl, hdl
        
        Returns:
            Dictionary containing crisp output and fuzzy memberships
        """
        
        # Step 1: Fuzzification
        fuzzified = {}
        for var_name, value in inputs.items():
            if var_name in self.variables:
                fuzzified[var_name] = self.variables[var_name].fuzzify(value)
        
        # Debug: Print fuzzified inputs
        print(f"Fuzzified inputs:")
        for var, memberships in fuzzified.items():
            print(f"  {var}: {memberships}")
        
        # Step 2: Rule evaluation
        rule_outputs = {"low": [], "moderate": [], "high": []}
        active_rules = []
        
        for i, rule in enumerate(self.rules):
            strength, conclusion = rule.evaluate(fuzzified)
            if strength > 0:
                risk_level = conclusion["risk"]
                rule_outputs[risk_level].append(strength)
                active_rules.append((i, strength, conclusion))
        
        # Debug: Print active rules
        print(f"Active rules: {len(active_rules)}")
        for rule_idx, strength, conclusion in active_rules:
            print(f"  Rule {rule_idx}: strength={strength:.3f}, conclusion={conclusion}")
        
        # Step 3: Aggregation (take maximum for each output set)
        aggregated = {}
        for risk_level, strengths in rule_outputs.items():
            aggregated[risk_level] = max(strengths) if strengths else 0.0
        
        # Step 4: Defuzzification using centroid method
        risk_sets = self.variables["risk"].sets
        numerator = 0.0
        denominator = 0.0
        
        # Sample points for centroid calculation
        x_points = np.linspace(0, 1, 1000)
        
        for x in x_points:
            membership_sum = 0.0
            for risk_level, strength in aggregated.items():
                if risk_level in risk_sets:
                    set_membership = risk_sets[risk_level].membership(x)
                    clipped_membership = min(set_membership, strength)
                    membership_sum = max(membership_sum, clipped_membership)
            
            numerator += x * membership_sum
            denominator += membership_sum
        
        crisp_output = numerator / denominator if denominator > 0 else 0.5
        
        return {
            "crisp_output": crisp_output,
            "fuzzy_memberships": aggregated,
            "fuzzified_inputs": fuzzified
        }
    
    def visualize_membership_functions(self, variable_name: str):
        """Visualize membership functions for a given variable"""
        if variable_name not in self.variables:
            print(f"Variable {variable_name} not found")
            return
        
        var = self.variables[variable_name]
        x_min, x_max = var.universe
        x = np.linspace(x_min, x_max, 1000)
        
        plt.figure(figsize=(10, 6))
        for set_name, fuzzy_set in var.sets.items():
            y = [fuzzy_set.membership(xi) for xi in x]
            plt.plot(x, y, label=set_name, linewidth=2)
        
        plt.xlabel(variable_name.capitalize())
        plt.ylabel('Membership Degree')
        plt.title(f'Membership Functions for {variable_name.capitalize()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        plt.show()

# Example usage and testing
def main():
    # Create FIS instance
    fis = HeartDiseaseFIS()
    
    # Test cases
    test_cases = [
        {
            "name": "Low Risk Patient",
            "inputs": {"age": 30, "trestbps": 120, "chol": 180, "fbs": 0, "ldl": 100, "hdl": 70}
        },
        {
            "name": "Moderate Risk Patient", 
            "inputs": {"age": 50, "trestbps": 140, "chol": 220, "fbs": 0.5, "ldl": 150, "hdl": 45}
        },
        {
            "name": "High Risk Patient",
            "inputs": {"age": 65, "trestbps": 170, "chol": 300, "fbs": 1, "ldl": 200, "hdl": 35}
        }
    ]
    
    print("Heart Disease Risk Prediction using Fuzzy Inference System")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(f"Inputs: {test_case['inputs']}")
        
        result = fis.predict(test_case['inputs'])
        
        print(f"Crisp Output (Risk Probability): {result['crisp_output']:.3f}")
        print(f"Fuzzy Memberships: {result['fuzzy_memberships']}")
        
        # Interpret result
        risk_prob = result['crisp_output']
        if risk_prob < 0.4:
            interpretation = "Low Risk"
        elif risk_prob < 0.7:
            interpretation = "Moderate Risk"
        else:
            interpretation = "High Risk"
        
        print(f"Interpretation: {interpretation}")
        print("-" * 40)
    
    # Visualize some membership functions
    print("\nGenerating membership function plots...")
    fis.visualize_membership_functions("age")
    fis.visualize_membership_functions("risk")

if __name__ == "__main__":
    main()