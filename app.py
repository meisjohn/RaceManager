from ast import Num
from flask import Flask, render_template, request, redirect, url_for
import itertools
import json
import os
from enum import Enum
import csv


app = Flask(__name__)

# Load patrol names from JSON config file
try:
    with open("patrol_config.json", "r") as f:
        patrol_names = json.load(f)
except FileNotFoundError:
    patrol_names = {
        1: "Foxes", 
        2: "Hawks", 
        3: "Mountain Lions", 
        4: "Navgators", 
        5: "Adventurers", 
        "Exhibition": "Exhibition"
    }

NUM_LANES = 4

class Rounds(Enum):
    NONE = 0
    FIRST = 1
    SEMI = 2
    FINAL = 3

class Participant:
    def __init__(self, first_name, last_name, patrol):
        self.first_name = first_name
        self.last_name = last_name
        self.patrol = patrol
        self.car_weight_oz = 0
        self.times = []
        self.average_time = 0
        self.best_time = 0
        self.car_name = None
    def __str__(self):
        return f"P({self.car_name}): {self.patrol},{self.first_name} {self.last_name}, {self.car_weight_oz}"

class Race:
    def __init__(self):
        self.race_number = assign_race_number()
        self.heats = []
        self.round = Rounds.NONE
    def __str__(self):
        my_str=f"R({self.race_number}): Round: {self.round}" + os.linesep
        my_str+=os.linesep.join([f"Heat: {h}" for h in self.heats])
        return my_str

class Heat:
    def __init__(self, heat_number):
        self.heat_id = assign_heat_index()
        self.heat_number = heat_number
        self.lanes = {}
        self.times = {}
    def __str__(self):
        my_str=f"H({self.heat_id} {self.heat_number}):" + os.linesep
        my_str+=os.linesep.join([f"Lane {l}: {p}" for l,p in sorted(self.lanes.items())])
        return my_str

# Race Data
participants = []
races = []
initial_races_completed = {}
semi_final_races_completed = {}
race_index = 0
heat_index = 0

def assign_race_number():
    global race_index
    race_index += 1
    return race_index

def assign_heat_index():
    global heat_index
    heat_index += 1
    return heat_index

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form["name"]
        try:
            patrol = int(request.form["patrol"])
            if 1 <= patrol <= len(patrol_names):
                add_participant(name, patrol)
                return redirect(url_for("index"))
            else:
                return render_template("index.html", 
                                       participants=participants, 
                                       error=f"patrol must be between 1 and {len(patrol_names)}.")
        except ValueError:
            return render_template("index.html", 
                                   participants=participants, 
                                   error="Invalid patrol input.")
    return render_template("index.html", 
                           participants=participants, 
                           patrol_names=patrol_names)

def add_participant(first_name,last_name, patrol):
    new_p = Participant(first_name,last_name, patrol)

    num_patrol = len([p for p in participants if p.patrol == patrol])
    new_p.car_name = f"{patrol[0]}{num_patrol}"
    participants.append(new_p)

def weigh_car(car_name, weight):
    if car_name is not None:
        p = [p for p in participants if p.car_name == car_name]
        if len(p) == 1:
            p[0].car_weight_oz = weight

@app.route("/schedule_initial")
def schedule_initial():
    schedule_initial_races()
    return redirect(url_for("schedule")) # Redirect to the main schedule page

def clear_races(round: Rounds = Rounds.NONE):
    if round == Rounds.NONE:
        races.clear()
    else:
        for r in races:
            if r.round == round:
                del r

def schedule_initial_races():

    for patrol in patrol_names:
        patrol_racers = [p for p in participants if p.patrol == patrol]

        race_groups = group_racers(patrol_racers)

        assign_paired_lanes(race_groups, Rounds.FIRST)

        initial_races_completed[patrol] = False # Initialize to False at the start

def group_racers(racers):
    groups = []
    num_racers = len(racers)
    if num_racers <= NUM_LANES:
        groups = [racers + [None] * (NUM_LANES - num_racers)]
    else:
        # Get our total number of races
        num_races = (num_racers + NUM_LANES - 1) // NUM_LANES
        # Minimum number of cars per race
        base_racers_per_race = num_racers // num_races
        remainder = num_racers % num_races
        distribution = [base_racers_per_race + 1] * remainder + \
                [base_racers_per_race] * (num_races - remainder)
        racer_idx = 0
        for d in distribution:
            race = racers[racer_idx:racer_idx+d] + [None] * (NUM_LANES-d)
            racer_idx += d
            groups.append(race)
    return groups

def assign_paired_lanes(groups, round: Rounds):
    global races
    lanes_half_a = [i for i in range(0,int(NUM_LANES/2))]
    lanes_half_b = [i for i in range(int(NUM_LANES/2),NUM_LANES)]
    swapped_lanes = list(reversed(lanes_half_a)) + list(reversed(lanes_half_b))
    # Create combinations and assign lanes:
    for grp in groups:
        race = Race()
        race.round = round
        # Create two heats, swapping lanes on the second heat
        for heat_num in [1, 2]:
            heat = Heat(heat_num)
            # Then assign lanes normally for each half for the first heat
            # Then reassign them in half-reverse for the second heat
            # i.e. 0,1,2,3 then 1,0,3,2 for 4 lanes
            if heat_num == 1:
                for lane in range(NUM_LANES):
                    heat.lanes[lane] = grp[lane]
            else:
                for lane in range(NUM_LANES):
                    heat.lanes[swapped_lanes[lane]]=grp[lane]
            race.heats.append(heat)
        races.append(race)

def assign_all_lanes(race_group, round: Rounds):
    global races
    race = Race()
    race.round = round
    if len(race_group) < NUM_LANES:
        for i in range(len(race_group), NUM_LANES):
            race_group.append(None)
    num_cars = len(race_group)
    # Assigning Lanes:
    # Goal is to have every car run in every lane    
    # If more cars run than lanes, then a different
    # car will sit out each race.
    for heat_idx in range(num_cars):
        heat = Heat(heat_idx)
        for lane in range (NUM_LANES):
            heat.lanes[lane] = race_group[(heat_idx + lane) % num_cars]
        race.heats.append(heat)
    races.append(race)

@app.route("/complete_initial/<int:patrol>") # New route to mark initial races as completed
def complete_initial(patrol):
    initial_races_completed[patrol] = True
    return redirect(url_for("schedule"))

@app.route("/schedule_semifinal") # New route for scheduling semi-final races
def schedule_semifinal():
    schedule_semi_final_races()
    return redirect(url_for("schedule"))

def schedule_semi_final_races():

    for patrol in patrol_names:
        if patrol != "Exhibition" and initial_races_completed.get(patrol, False): # Only schedule if initial races are completed
            # Get top racers
            patrol_participants = [p for p in participants if p.patrol == patrol]
            patrol_participants = sorted(patrol_participants, 
                                         key=lambda p: (p.average_time, p.best_time))[:NUM_LANES]

            if num_participants >= 3:
                pass #TODO


            semi_final_races_completed[patrol] = False # Initialize to False at the start

@app.route("/complete_semifinal/<int:patrol>") # New route to mark semi-final races as completed
def complete_semifinal(patrol):
    semi_final_races_completed[patrol] = True
    return redirect(url_for("schedule"))

@app.route("/schedule_final")  # New route for scheduling final races
def schedule_final():
    schedule_final_races()
    return redirect(url_for("schedule"))

def schedule_final_races():
   # ... (Logic to clear existing final races, if any)
    race_number = len(races) + 1  # Start heat numbers after other races

    fastest_racers = []
    for patrol in patrol_names:
        if patrol != "Exhibition" and semi_final_races_completed.get(patrol, False): # Only schedule if semi-final races are completed
            patrol_participants = [p for p in participants if p.patrol == patrol]
            if patrol_participants:
                 fastest = sorted(patrol_participants, key=lambda p: (p.average_time, p.best_time))[0]
                 fastest_racers.append(fastest)

    if len(fastest_racers) > 1:
        final_race = Race()
        for i, racer in enumerate(fastest_racers):
            final_race.lanes[i+1] = racer
        races.append(final_race)


@app.route("/schedule") # Main schedule page
def schedule():
    return render_template("schedule.html", races=races, patrol_names=patrol_names,
                           initial_races_completed=initial_races_completed,
                           semi_final_races_completed=semi_final_races_completed)

@app.route("/results", methods=["GET", "POST"])
def results():
    if request.method == "POST":
        for race in races:
            for heat in race.heats:
                for lane in range(NUM_LANES):
                    participant = race.lanes.get(lane)
                    if participant:
                        try:
                            time_key = f"time_race_{race.race_number}_heat_{heat.heat_number}_lane_{lane}"
                            time = float(request.form.get(time_key, 0))
                            race.times[lane] = time
                            participant.times.append(time)
                            participant.best_time = min(participant.times) if participant.times else 0 # Track best time
                        except ValueError:
                            return render_template("results_input.html",
                                                   races=races,
                                                   patrol_names=patrol_names,
                                                   error=f"Invalid time input for Heat {race.race_number}, Lane {lane}.")
            calculate_averages()
        return redirect(url_for("display_results"))
    return render_template("results_input.html", races=races, patrol_names=patrol_names)

def calculate_averages():
    for p in participants:
        if p.times:
            p.average_time = sum(p.times) / len(p.times)
        else:
            p.average_time = 0

@app.route("/display_results")
def display_results():
    sorted_participants = sorted(participants, key=lambda x: (x.average_time, x.best_time))
    return render_template("results.html", participants=sorted_participants, patrol_names=patrol_names)


def load_roster(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            add_participant(row["First Name"], row["Last Name"], row["Patrol"])



if __name__ == "__main__":
    app.run(debug=True)
