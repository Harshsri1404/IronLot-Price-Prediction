# agent_Harsh.py
# bidding agent for the IronLot auction simulation
# loads trained model and encoders, predicts car value, places bids

import pickle
import numpy as np


class BiddingAgent:

    def __init__(self):
        # loading model and encoders once at startup
        # dont want to reload pkl files for every single car, that would be slow
        with open('model_Harsh.pkl', 'rb') as f:
            bundle = pickle.load(f)

        self.model = bundle['model']
        self.feature_cols = bundle['features']
        self.current_year = bundle['current_year']  # 2015

        with open('encoders_Harsh.pkl', 'rb') as f:
            self.encoders = pickle.load(f)

        self.cat_cols = ['make', 'model', 'trim', 'body', 'transmission',
                         'color', 'interior', 'state']

        # tracking budget and performance
        self.bankroll = 500000
        self.wins = 0
        self.total_spent = 0
        self.total_value = 0


    def analyze_item(self, item: dict) -> float:
        # takes one car as a dict, returns predicted selling price
        # everything has to work on a single row, no df operations allowed

        # --- cleaning ---
        # using try/except everywhere because real auction data can have garbage values

        try:
            year = int(item.get('year', self.current_year))
        except (ValueError, TypeError):
            year = self.current_year

        try:
            odometer = float(item.get('odometer', 60000))
            # 300k+ is sensor glitch on old cars, negative doesnt make sense
            if odometer > 300000 or odometer < 0:
                odometer = 60000
        except (ValueError, TypeError):
            odometer = 60000

        try:
            condition = float(item.get('condition', 3.0))
            condition = max(1.0, min(5.0, condition))  # condition scale is 1-5
        except (ValueError, TypeError):
            condition = 3.0

        # --- feature engineering (same as training) ---
        car_age = max(0, self.current_year - year)
        odo_per_year = odometer / max(car_age, 0.5)  # max to avoid divide by zero
        condition_x_age = condition * car_age
        low_odo_good_cond = int(odometer < 30000) * condition

        # --- encoding categorical columns ---
        enc_vals = {}
        for col in self.cat_cols:
            # handling missing/None values
            raw = str(item.get(col, 'Unknown') or 'Unknown').strip()
            le = self.encoders[col]

            if raw in le.classes_:
                # normal case, value was seen in training data
                enc_vals[col + '_enc'] = le.transform([raw])[0]
            elif 'Unknown' in le.classes_:
                # unseen value (new car brand etc), treat as Unknown
                enc_vals[col + '_enc'] = le.transform(['Unknown'])[0]
            else:
                # shouldnt happen but just in case
                enc_vals[col + '_enc'] = 0

        # --- assembling feature vector ---
        # order matters! has to match exactly what model was trained on
        raw_features = {
            'year': year,
            'car_age': car_age,
            'odometer': odometer,
            'odo_per_year': odo_per_year,
            'condition': condition,
            'condition_x_age': condition_x_age,
            'low_odo_good_cond': low_odo_good_cond,
            **enc_vals
        }

        X = np.array([[raw_features[f] for f in self.feature_cols]])
        predicted = float(self.model.predict(X)[0])

        # minimum floor of $100, model shouldnt predict negative but just in case
        return max(predicted, 100.0)


    def place_bid(self, predicted_value, bankroll, current_highest_bid, round_number):
        # decides how much to bid based on predicted value, budget, competition, round
        # returns 0 to pass on a car

        V = predicted_value
        B = bankroll
        H = current_highest_bid
        R = round_number

        # how much below fair value are we willing to pay
        # early rounds be more aggressive to actually win cars
        # late rounds tighten up to protect profit margin
        if R <= 3:
            discount = 0.93
        elif R <= 6:
            discount = 0.90
        else:
            discount = 0.87

        max_bid = V * discount

        # never put more than 20% of budget on one car
        # one bad prediction shouldnt wipe out the whole bankroll
        budget_cap = B * 0.20
        max_bid = min(max_bid, budget_cap)

        # if current bid is already above what we'd pay, just skip this car
        if max_bid <= H:
            return 0.0

        # how much of the remaining gap to jump in one bid
        # aggressive early = close deal fast
        # conservative late = let rivals overpay and drop out
        gap = max_bid - H
        if R <= 3:
            aggression = 0.50
        elif R <= 6:
            aggression = 0.35
        else:
            aggression = 0.20

        MIN_INCREMENT = 100.0
        increment = max(MIN_INCREMENT, gap * aggression)
        bid = H + increment

        # making sure we dont go over our max
        bid = min(bid, max_bid)

        # rounding to nearest 50, looks more realistic
        bid = round(bid / 50) * 50

        # after rounding make sure we're still above current bid
        bid = max(bid, H + MIN_INCREMENT)

        # final check, dont spend more than we have
        if bid > B:
            return 0.0

        return float(bid)


# the arena calls these two functions directly
# keeping one global agent so we dont reload model every time

_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        _agent = BiddingAgent()
    return _agent

def analyze_item(item: dict) -> float:
    return _get_agent().analyze_item(item)

def place_bid(predicted_value, bankroll, current_highest_bid, round_number):
    return _get_agent().place_bid(predicted_value, bankroll, current_highest_bid, round_number)


# quick test when running directly
if __name__ == '__main__':
    test_car = {
        'year': 2012,
        'make': 'Honda',
        'model': 'Civic',
        'trim': 'LX',
        'body': 'Sedan',
        'transmission': 'automatic',
        'state': 'CA',
        'condition': 3.5,
        'odometer': 45000,
        'color': 'gray',
        'interior': 'gray'
    }

    pred = analyze_item(test_car)
    print(f"predicted value: ${pred:,.0f}")

    bid1 = place_bid(pred, 500000, pred * 0.70, round_number=2)
    print(f"bid at round 2 when H is 70% of value: ${bid1:,.0f}")

    bid2 = place_bid(pred, 500000, pred * 0.92, round_number=7)
    print(f"bid at round 7 when H is near ceiling: ${bid2:,.0f}")
