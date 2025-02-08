from flask import Flask, render_template, request, redirect, url_for
import itertools
import json
from enum import Enum


app = Flask(__name__)

# Load subgroup names from JSON config file
try:
    with open("subgroup_config.json", "r") as f:
        subgroup_names = json.load(f)
except FileNotFoundError:
    subgroup_names = {
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
    def __init__(self, name, subgroup):
        self.name = name
        self.subgroup = subgroup
        self.car_number = None
        self.times = []
        self.average_time = 0
        self.car_name = None

class Race:
    def __init__(self, heat_number):
        self.heat_number = heat_number
        self.lanes = {}
        self.times = {}
        self.round = Round.NONE

participants = []
races = []

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form["name"]
        try:
            subgroup = int(request.form["subgroup"])
            if 1 <= subgroup <= len(subgroup_names):
                add_participant(name, subgroup)
                return redirect(url_for("index"))
            else:
                return render_template("index.html", 
                                       participants=participants, 
                                       error=f"Subgroup must be between 1 and {len(subgroup_names)}.")
        except ValueError:
            return render_template("index.html", 
                                   participants=participants, 
                                   error="Invalid subgroup input.")
    return render_template("index.html", 
                           participants=participants, 
                           subgroup_names=subgroup_names)

def add_participant(name, subgroup):
    participants.append(Participant(name, subgroup))
    num_participants = len([p for p in participants if p.subgroup == subgroup])
    participants[-1].car_number = num_participants
    participants[-1].car_name = f"{subgroup[0]}{num_participants}"

@app.route("/schedule_initial")
def schedule_initial():
    schedule_initial_races()
    return redirect(url_for("schedule")) # Redirect to the main schedule page

def schedule_initial_races():
    races.clear() # Clear any existing races
    heat_number = 1

    for subgroup in subgroup_names:
        subgroup_participants = [p for p in participants if p.subgroup == subgroup]
        num_participants = len(subgroup_participants)
        full_groups = num_participants // NUM_LANES
        remainder = num_participants & NUM_LANES

        if num_participants =< NUM_LANES:
            race_groups = subgroup_participants
        else:

            # Assign all races but last two
            race_groups = [subgroup_participants[i:i+NUM_LANES]
                           for i in range(0, (full_groups-1)*NUM_LANES, NUM_LANES)]
            resume_idx = ((full_groups-1) * NUM_LANES) + 1

            if remainder != 0 && remainder < NUM_LANES/2:
                # Rebalance
                last_row = int((NUM_LANES + remainder) / 2)
                penultimate_row = (NUM_LANES + remainder) - last_row
                race_groups += subgroup_participants[resume_idx:resume_index+penultimate_row]
                race_groups += subgroup_participants[resume_index+penultimate_row:-1]

            else:
                # Take it as-is
                race_groups += [subgroup_participants[i:i+chunk]
                                for i in range(resume_idx, num_participants, NUM_LANES)]
            



        # Create combinations and assign lanes:
        for grp in race_groups:
                pairs = [grp[i:i+1] for i in range(0,len(grp),2)]

                # Create two heats, swapping lanes
                for heat_num in [1, 2]:
                    race = Race(heat_number)
                    race.round = Rounds.FIRST
                    if heat_num == 1:
                        for lane in range(0,NUM_LANES,2):
                            if (lane/2) < len(pairs):
                                race.lanes[lane] = pairs[lane/2][0]
                                race.lanes[lane+1] = pairs[lane/2][1] if len (pairs[lane/2]) == 2 else None
                            else:
                                race.lanes[lane] = None
                                race.lanes[lane+1] = None


                    else:
                        for lane in range(0,NUM_LANES,2):
                            if (lane/2) < len(pairs):
                                race.lanes[lane+1] = pairs[lane/2][0]
                                race.lanes[lane] = pairs[lane/2][1] if len (pairs[lane/2]) == 2 else None
                            else:
                                race.lanes[lane] = None
                                race.lanes[lane+1] = None


                    races.append(race)

        initial_races_completed[subgroup] = False # Initialize to False at the start

@app.route("/complete_initial/<int:subgroup>") # New route to mark initial races as completed
def complete_initial(subgroup):
    initial_races_completed[subgroup] = True
    return redirect(url_for("schedule"))

@app.route("/schedule_semifinal") # New route for scheduling semi-final races
def schedule_semifinal():
    schedule_semi_final_races()
    return redirect(url_for("schedule"))

def schedule_semi_final_races():
    # ... (Logic to clear existing semi-final races, if any)
    heat_number = len(races) + 1  # Start heat numbers after initial races

    for subgroup in subgroup_names:
        if subgroup != "Exhibition" and initial_races_completed.get(subgroup, False): # Only schedule if initial races are completed
            subgroup_participants = [p for p in participants if p.subgroup == subgroup]
            subgroup_participants = sorted(subgroup_participants, 
                                           key=lambda p: (p.average_time, p.best_time))[:4] # Top 4
            num_participants = len(subgroup_participants)

            if num_participants >= 3:


            semi_final_races_completed[subgroup] = False # Initialize to False at the start

@app.route("/complete_semifinal/<int:subgroup>") # New route to mark semi-final races as completed
def complete_semifinal(subgroup):
    semi_final_races_completed[subgroup] = True
    return redirect(url_for("schedule"))

@app.route("/schedule_final")  # New route for scheduling final races
def schedule_final():
    schedule_final_races()
    return redirect(url_for("schedule"))

def schedule_final_races():
   # ... (Logic to clear existing final races, if any)
    heat_number = len(races) + 1  # Start heat numbers after other races

    fastest_racers = []
    for subgroup in subgroup_names:
        if subgroup != "Exhibition" and semi_final_races_completed.get(subgroup, False): # Only schedule if semi-final races are completed
            subgroup_participants = [p for p in participants if p.subgroup == subgroup]
            if subgroup_participants:
                 fastest = sorted(subgroup_participants, key=lambda p: (p.average_time, p.best_time))[0]
                 fastest_racers.append(fastest)

    if len(fastest_racers) > 1:
        final_race = Race(heat_number)
        for i, racer in enumerate(fastest_racers):
            final_race.lanes[i+1] = racer
        races.append(final_race)


@app.route("/schedule") # Main schedule page
def schedule():
    return render_template("schedule.html", races=races, subgroup_names=subgroup_names,
                           initial_races_completed=initial_races_completed,
                           semi_final_races_completed=semi_final_races_completed)

@app.route("/results", methods=["GET", "POST"])
def results():
    if request.method == "POST":
        for race in races:
            for lane in range(1, 5):  # Up to 4 lanes
                participant = race.lanes.get(lane)
                if participant:
                    try:
                        time_key = f"time_race_{race.heat_number}_lane_{lane}"
                        time = float(request.form.get(time_key, 0))
                        race.times[lane] = time
                        participant.times.append(time)
                        participant.best_time = min(participant.times) if participant.times else 0 # Track best time
                    except ValueError:
                        return render_template("results_input.html", races=races, subgroup_names=subgroup_names, error=f"Invalid time input for Heat {race.heat_number}, Lane {lane}.")
        calculate_averages()
        return redirect(url_for("display_results"))
    return render_template("results_input.html", races=races, subgroup_names=subgroup_names)

def calculate_averages():
    for p in participants:
        if p.times:
            p.average_time = sum(p.times) / len(p.times)
        else:
            p.average_time = 0

@app.route("/display_results")
def display_results():
    sorted_participants = sorted(participants, key=lambda x: (x.average_time, x.best_time))
    return render_template("results.html", participants=sorted_participants, subgroup_names=subgroup_names)

if __name__ == "__main__":
    app.run(debug=True)
