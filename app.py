from ast import Num
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, flash
import itertools
import json
import os
from enum import Enum
import csv
from io import StringIO
import traceback
import uuid
from collections import defaultdict
import secrets


app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(16)

# Load patrol names from JSON config file
try:
    with open("patrol_config.json", "r") as f:
        patrol_names = json.load(f)
except FileNotFoundError:
    patrol_names = {
        "1": "Foxes", 
        "2": "Hawks", 
        "3": "Mountain Lions", 
        "4": "Navgators", 
        "5": "Adventurers", 
        "Exhibition": "Exhibition"
    }

NUM_LANES = 4

DATA_FILE = "race_data.json"  # Define the filename for storing data

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
        self.best_time = float('inf')
        self.best_time_race_number = None
        self.car_name = None
        self.car_number = None
        self.participant_id = uuid.uuid4().hex
    def __str__(self):
        return f"P({self.car_name}): {self.patrol},{self.first_name} {self.last_name}, {self.car_weight_oz}"
    def toJSON(self):
        return self.__dict__

class Race:
    def __init__(self, patrol, race_number):
        self.patrol = patrol
        self.race_number = race_number
        self.heats = []
        self.round = Rounds.NONE
    def __str__(self):
        my_str=f"R({patrol_names[self.patrol]} {self.race_number}): Round: {self.round}" + os.linesep
        my_str+=os.linesep.join([f"Heat: {h}" for h in self.heats])
        return my_str
    def toJSON(self):
        return { "patrol": self.patrol,
                 "race_number": self.race_number,
                 "round": self.round.value,
                 "heats": [ h.toJSON() for h in self.heats]
                }

class Heat:
    def __init__(self, heat_number):
        self.heat_id = uuid.uuid4().hex
        self.heat_number = heat_number
        self.lanes = {}
        self.times = {}
    def __str__(self):
        my_str=f"H({self.heat_id} {self.heat_number}):" + os.linesep
        my_str+=os.linesep.join([f"Lane {l}: {p}" for l,p in sorted(self.lanes.items())])
        return my_str
    def toJSON(self):
        return {
                "heat_id": self.heat_id,
                "heat_number": self.heat_number,
                "lanes": {lane: p.participant_id if p else None for lane, p in self.lanes.items()},
                "times": self.times
                }

class Design:
    def __init__(self, participant):
        self.participant = participant
        self.scores = {}  # {judge_id: score}

    def toJSON(self):
        return {
            "participant_id": self.participant.participant_id,
            "scores": self.scores,
        }

# Race Data
participants = []
races = []
initial_races_completed = {p:False for p in patrol_names}
semi_final_races_completed = {p:False for p in patrol_names if p != "Exhibition"}
designs = []
judging_active = True

def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            # Reconstruct participants, races, etc. from loaded data
            global participants, races, initial_races_completed, semi_final_races_completed
            global designs

            participants = []
            for p_data in data.get("participants", []):
                p = Participant(p_data["first_name"], p_data["last_name"], p_data["patrol"])
                p.__dict__.update(p_data)  # Update all other attributes
                participants.append(p)

            races = []
            for r_data in data.get("races", []):
                r = Race(r_data["patrol"], int(r_data["race_number"]))
                r.__dict__.update(r_data)
                r.round=Rounds(int(r_data["round"]))
                r.heats = []
                for h_data in r_data.get("heats", []):
                    h = Heat(int(h_data["heat_number"]))
                    h.__dict__.update(h_data)
                    h.lanes = {}
                    for lane_num_str, p_id in h_data.get("lanes", {}).items():
                        lane_num = int(lane_num_str)
                        if p_id:
                            p = next((p for p in participants if p.participant_id == p_id), None)
                            h.lanes[lane_num] = p
                        else:
                            h.lanes[lane_num] = None
                    h.times = {int(k): float(t) for k, t in h_data.get("times", {}).items()}
                    r.heats.append(h)
                races.append(r)

            designs = []
            for d_data in data.get("designs", []):
                p_id = d_data["participant_id"]
                participant = next((p for p in participants if p.participant_id == p_id), None)
                if participant:
                    d = Design(participant)
                    d.scores = d_data.get("scores", {}) # Load scores
                    designs.append(d)


            initial_races_completed = data.get("initial_races_completed", {})
            semi_final_races_completed = data.get("semi_final_races_completed", {})

    except FileNotFoundError:
        pass  # Start with empty data if the file doesn't exist

def save_data():
    data = {
        "participants": [p.toJSON() for p in participants],
        "races": [r.toJSON() for r in races],
        "designs": [d.toJSON() for d in designs],
        "initial_races_completed": initial_races_completed,
        "semi_final_races_completed": semi_final_races_completed,
    }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, default=str)  # Use default=str to handle non-serializable objects like Enums

# Load data when the app starts
try:
    load_data()
    if participants: # Only initialize if there are participants
        save = False
        for p in participants:
            if p not in [d.participant for d in designs]:
                designs.append(Design(p))
                save = True
        if save:
            save_data() # Save the designs
except Exception as e:
    print(f"Encountered exception loading saved data")
    traceback.print_exc()

@app.route("/", methods=["GET", "POST"])
def index():
    error = None  # Initialize error variable
    if request.method == "POST":
        first_name = request.form.get("first_name") # Get first name
        last_name = request.form.get("last_name")  # Get last name
        try:
            patrol = request.form.get("patrol") # Get patrol
            if patrol in patrol_names:
                new_participant = add_participant(first_name, last_name, patrol)
                save_data() # Save data after adding participant
                return redirect(url_for("index"))  # Redirect after successful addition
            else:
                error = f"Patrol {patrol} is invalid."  # Set error message
        except ValueError as e:
            error = str(e)  # Set error message

    return render_template("index.html", 
                           participants=participants, 
                           patrol_names=patrol_names,
                           error=error)  # Pass error to the template

@app.route("/edit_participant/<int:participant_id>", methods=["GET", "POST"])
def edit_participant(participant_id):
    participant = next((p for p in participants if p.car_number == participant_id), None) # Find by car_number
    if not participant:
        return "Participant not found", 404

    if request.method == "POST":
        try:
            car_weight_oz = float(request.form.get("car_weight_oz"))
            car_name = request.form.get("car_name")

            participant.car_weight_oz = car_weight_oz
            participant.car_name = car_name

            save_data() # Save data after editing participant
            return redirect(url_for("index"))  # Redirect back to the participant list
        except ValueError:
            return "Invalid weight input. Please enter a number.", 400

    return render_template("edit_participant.html", participant=participant, patrol_names = patrol_names)

@app.route("/delete_participant/<int:participant_id>", methods=["POST"])
def delete_participant(participant_id):
    participant = next((p for p in participants if p.car_number == participant_id), None)
    if not participant:
        return "Participant not found", 404

    participants.remove(participant)  # Remove the participant

    save_data() # Save data after removing participant
    return redirect(url_for("index"))  # Redirect back to the participant list

@app.route("/participant_times/<int:participant_id>")
def participant_times(participant_id):
    participant = next((p for p in participants if p.car_number == participant_id), None)
    if not participant:
        return "Participant not found", 404
    return render_template("participant_times.html", participant=participant, patrol_names=patrol_names)


def add_participant(first_name,last_name, patrol):

    global participants, designs

    if patrol in patrol_names.values():
        for k,v in patrol_names.items():
            if v == patrol:
                patrol = k
                break

    patrol = str(patrol)

    if patrol in patrol_names:

        new_p = Participant(first_name,last_name, patrol)

        existing_car_numbers = [p.car_number for p in participants if p.patrol == patrol]

        if existing_car_numbers:
            next_car_number = max(existing_car_numbers) + 1  # Find the maximum and add 1
        else:
            next_car_number = 1  # Start at 1 if no existing numbers

        new_p.car_number = next_car_number
        new_p.car_name = f"{patrol_names.get(patrol)[:1]}{next_car_number}"

        participants.append(new_p)

        designs.append(Design(p))

        return new_p
    else:
        raise ValueError(f"Invalid Patrol: {patrol}")

def clear_races(round: Rounds = Rounds.NONE):
    global races
    races.clear()
    # If races have been rescheduled, then all data needs to be reset
    for p in participants:
        p.times = []
        p.average_time = 0
        p.best_time = float('inf')
        p.best_time_race_number = None

def schedule_initial_races():

    race_number = 1
    for patrol in patrol_names:
        patrol_racers = [p for p in participants if p.patrol == patrol]

        if patrol_racers:
            race_groups = group_racers(patrol_racers)
            assign_paired_lanes(race_groups, Rounds.FIRST, race_number)
            race_number += len(race_groups)

            initial_races_completed[patrol] = False # Initialize to False at the start
            semi_final_races_completed[patrol] = False 
        else:
            initial_races_completed[patrol] = True # No races for this patrol
            semi_final_races_completed[patrol] = True # No races for this patrol


def schedule_semi_final_races(patrol):
    global races

    if patrol == "Exhibition":
        return  # Don't schedule semi-finals for the exhibition patrol
    if not initial_races_completed[patrol]:
        return  # Don't schedule semi's if we havent finished the races

    top_racers,_ = get_top_racers(Rounds.FIRST, patrol, NUM_LANES)

    if top_racers:
        race_groups = [top_racers]  # Create a single group of top racers
        assign_paired_lanes(race_groups, Rounds.SEMI, len(races) + 1) 

def schedule_final_races():
    global races

    # 1. Get Top Racers from Semi-Finals:
    top_racers = []
    for patrol in patrol_names:
        if patrol != "Exhibition" and semi_final_races_completed.get(patrol, False):  # Check if semi-finals are complete
            top_racer,_ = get_top_racers(Rounds.SEMI, patrol, 1) # Only want to get top 1
            if top_racer:
                top_racers.append(top_racer[0])

    # 2. Create Final Race (using all lanes assignment):
    if top_racers: # Only if there are top racers
        race_groups = [top_racers]
        assign_all_lanes(race_groups, Rounds.FINAL, len(races) + 1)  # Use assign_all_lanes

def get_top_racers(round: Rounds, patrol = None, racer_count=NUM_LANES):
    global races
    top_racers = []
    overall_racer_averages = {}
    if patrol is not None and patrol != "":
        filtered_races = [r for r in races if r.round == round and r.patrol == patrol]
    else:
        filtered_races = [r for r in races if r.round == round]
    if filtered_races: # Only if semi-final races exist for this patrol
        all_racer_averages = {}
        for race in filtered_races:
            race_averages = calculate_race_averages(race)
            for racer, avg_time in race_averages.items():
                if racer not in all_racer_averages:
                  all_racer_averages[racer] = []
                all_racer_averages[racer].append(avg_time)

        # Calculate overall average time and sort
        for racer, avg_times in all_racer_averages.items():
            overall_racer_averages[racer] = sum(avg_times) / len(avg_times) if avg_times else float('inf')

        sorted_racers = sorted(overall_racer_averages.items(), key=lambda item: item[1])
        top_racers = [racer for racer, avg_time in sorted_racers[:racer_count] 
                      if avg_time != float('inf')]
    return top_racers, overall_racer_averages


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

def assign_paired_lanes(groups, round: Rounds, race_number_start):
    global races
    lanes_half_a = [i for i in range(0,int(NUM_LANES/2))]
    lanes_half_b = [i for i in range(int(NUM_LANES/2),NUM_LANES)]
    swapped_lanes = list(reversed(lanes_half_a)) + list(reversed(lanes_half_b))
    # Create combinations and assign lanes:
    for grp in groups:
        race = Race(grp[0].patrol if grp[0] else None, race_number_start)
        race_number_start += 1
        race.round = round
        # Create two heats, swapping lanes on the second heat
        for heat_num in [1, 2]:
            heat = Heat(heat_num)
            # Then assign lanes normally for each half for the first heat
            # Then reassign them in half-reverse for the second heat
            # i.e. 0,1,2,3 then 1,0,3,2 for 4 lanes
            if heat_num == 1:
                for lane in range(NUM_LANES):
                    heat.lanes[lane+1] = grp[lane] if grp[lane] else None
            else:
                for lane in range(NUM_LANES):
                    heat.lanes[swapped_lanes[lane]+1]=grp[lane] if grp[lane] else None
            race.heats.append(heat)
        races.append(race)


def assign_all_lanes(race_group, round: Rounds, race_number_start):
    global races
    for grp in race_group: # Iterate through race groups (only one in the finals)
        race = Race(grp[0].patrol if grp[0] else None, race_number_start) # Assign patrol and race number
        race_number_start += 1
        race.round = round
        if len(grp) < NUM_LANES:
            for i in range(len(grp), NUM_LANES):
                grp.append(None) # Pad with None if fewer racers than lanes
        num_cars = len(grp)
        # Assigning Lanes:
        # Goal is to have every car run in every lane    
        # If more cars run than lanes, then a different
        # car will sit out each race.
        for heat_idx in range(num_cars):
            heat = Heat(heat_idx + 1) # Heat numbers start at 1
            for lane in range (NUM_LANES):
                heat.lanes[lane+1] = grp[(heat_idx + lane) % num_cars] if grp[(heat_idx + lane) % num_cars] else None # Lane numbers start at 1
            race.heats.append(heat)
        races.append(race)

@app.route("/enter_times/<int:race_number>/<int:heat_number>", methods=["GET", "POST"])
def enter_times(race_number, heat_number):
    race = next((r for r in races if r.race_number == race_number), None)
    if not race:
        return "Race not found", 404
    heat = next((h for h in race.heats if h.heat_number == heat_number), None)
    if not heat:
        return "Heat not found", 404

    if request.method == "POST":
      for lane in range(1, NUM_LANES + 1):
          time_key = f"time_race_{race_number}_heat_{heat_number}_lane_{lane}"
          time = request.form.get(time_key, None)
          if time is not None:
              try:
                  time = float(time)
                  heat.times[lane] = time
                  participant = heat.lanes.get(lane)
                  if participant:
                      participant.times.append(time)
                      calculate_race_statistics(participant)
              except ValueError:
                  return "Invalid time input", 400
      save_data() # Saving data after entering race times
      return redirect(url_for("schedule", patrol=race.patrol, round=race.round.value)) # Add round parameter

    return render_template("enter_times.html", race=race, heat=heat, NUM_LANES=NUM_LANES)

@app.route("/schedule_initial")
def schedule_initial():
    clear_races()
    schedule_initial_races()
    save_data() # Saving data after scheduling initial round
    return redirect(url_for("schedule", round=Rounds.FIRST.value)) # Redirect to the main schedule page

@app.route("/complete_initial/<patrol>")
def complete_initial(patrol):
    initial_races_completed[patrol] = True
    return redirect(url_for("schedule", round=Rounds.FIRST.value)) # Add round parameter

@app.route("/schedule_semifinal/<patrol>")
def schedule_semifinal(patrol):
    schedule_semi_final_races(patrol)
    save_data() # Saving data after scheduling semi-final round
    return redirect(url_for("schedule", round=Rounds.SEMI.value)) # Add round parameter

@app.route("/complete_semifinal/<patrol>") # New route to mark semi-final races as completed
def complete_semifinal(patrol):
    semi_final_races_completed[patrol] = True
    return redirect(url_for("schedule", round=Rounds.SEMI.value)) # Add round parameter

@app.route("/schedule_final/<patrol>")
def schedule_final(patrol):
    schedule_final_races()
    save_data() # Saving data after scheduling final round
    return redirect(url_for("schedule", round=Rounds.FINAL.value)) # Add round parameter

@app.route("/schedule", methods=["GET"])
def schedule():
    selected_patrol = request.args.get("patrol", "")
    selected_round_str = request.args.get("round", str(Rounds.FIRST.value)) # Get round from query params
    try:
        selected_round = Rounds(int(selected_round_str))  # Convert to Rounds enum
    except ValueError:
        selected_round = Rounds.FIRST  # Default to first round if invalid round value

    selected_round_name = ""
    if selected_round == Rounds.FIRST:
        selected_round_name = "First Round"
    elif selected_round == Rounds.SEMI:
        selected_round_name = "Semi-Finals"
    elif selected_round == Rounds.FINAL:
        selected_round_name = "Finals"

    top_racers, overall_racer_averages = get_top_racers(selected_round, selected_patrol, NUM_LANES)

    if semi_final_races_completed:
        all_semi_final_races_complete = all(semi_final_races_completed.values())
    else:
        all_semi_final_races_complete = False

    return render_template("schedule.html", races=races, patrol_names=patrol_names,
                           selected_patrol=selected_patrol, 
                           selected_round=selected_round,
                           selected_round_value=selected_round.value,
                           selected_round_name = selected_round_name, 
                           Rounds=Rounds, NUM_LANES=NUM_LANES,
                           top_racers=list(enumerate(top_racers)),
                           overall_racer_averages=overall_racer_averages,
                           initial_races_completed=initial_races_completed,
                           semi_final_races_completed=semi_final_races_completed,
                           all_semi_final_races_complete=all_semi_final_races_complete )

def calculate_race_averages(race):
    racer_race_times = {}  # Store total times for racers in the race
    racer_heat_counts = {} # Store number of heats a racer participated in

    for heat in race.heats:
        for lane, racer in heat.lanes.items():
            if racer and heat.times.get(lane) is not None:
                if racer not in racer_race_times:
                    racer_race_times[racer] = 0
                    racer_heat_counts[racer] = 0
                racer_race_times[racer] += heat.times[lane]
                racer_heat_counts[racer] += 1
    
    racer_averages = {}
    for racer, total_time in racer_race_times.items():
        racer_averages[racer] = total_time / racer_heat_counts[racer] if racer_heat_counts[racer] > 0 else float('inf') # Handle cases where a racer might not have any times recorded

    return racer_averages

def calculate_race_statistics(participant):
    p = participant
    if p.times:
        p.best_time = min(p.times)
        p.average_time = sum(p.times) / len(p.times)
        p.best_time_race_number = get_best_time_race_number(p)
    else:
        p.best_time = float('inf')
        p.average_time = 0
        p.best_time_race_number = None

def get_best_time_race_number(participant):
    best_time_race_number = None
    if participant.times:
        if participant.best_time > 0:
            for race in races:
                for heat in race.heats:
                    for lane,p in heat.lanes.items():
                        if p == participant and lane in heat.times and \
                           heat.times[lane] == participant.best_time:
                               return race.race_number
    return None

@app.route("/judge_design", methods=["GET", "POST"])
def judge_design():
    global designs
    global judging_active
    if not judging_active:
        return "Judging is closed.", 403  # Or redirect, or display a message

    racers = {}
    for patrol in patrol_names:
        racers[patrol] = [p for p in participants if p.patrol == patrol]

    if request.method == "POST":
        judge_id = request.form.get("judge_id") # Get judge ID from the form
        if not judge_id: # Generate one if not provided
            judge_id = uuid.uuid4().hex

        for p in participants:
            rank = request.form.get(f"rank_{p.participant_id}")
            if rank:
                design = next((d for d in designs if d.participant == p), None)
                if design:
                    design.scores[judge_id] = int(rank)
        save_data()
        return redirect(url_for("design_results"))

    return render_template("judge_design.html", racers=racers, patrol_names=patrol_names)

@app.route("/close_judging")
def close_judging():
    global judging_active
    judging_active = False
    return redirect(url_for("design_results"))

@app.route("/open_judging")
def open_judging():
    global judging_active
    judging_active = True
    return redirect(url_for("design_results"))


def score_design(design):
    total_score = 0
    first = 0
    second = 0
    third = 0
    for judge_rank in design.scores.values():
        if judge_rank == 1:
            total_score += 3
            first += 1
        elif judge_rank == 2:
            total_score += 2
            second += 1
        elif judge_rank == 3:
            total_score += 1
            third += 1

    return (total_score,first,second,third)

@app.route("/design_results")
def design_results():

    sorted_scores_by_patrol = {}

    for patrol in patrol_names:
        patrol_scores = []
        for p in participants:
            if p.patrol == patrol:
                design = next((d for d in designs if d.participant == p), None)
                if design:
                    design_score = score_design(design)
                else:
                    design_score = (0,0,0,0)
                patrol_scores.append((p, design_score[0], design_score[1],
                                      design_score[2], design_score[3]))

        # Sort the scores by total score, #first votes, #second votes, #third votes
        patrol_scores.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4]))
        sorted_scores_by_patrol[patrol] = patrol_scores[:3] #Take the top 3
        

    return render_template("design_results.html", 
                           patrol_names=patrol_names, 
                           sorted_scores_by_patrol=sorted_scores_by_patrol,
                           judging_active=judging_active)


@app.route("/display_results")
def display_results():
    sorted_participants = sorted(participants, key=lambda x: (x.average_time, x.best_time))
    for p in sorted_participants:
        calculate_race_statistics(p)
    return render_template("results.html", 
                           participants=sorted_participants, 
                           patrol_names=patrol_names)

@app.route("/download_results")
def download_results():
    output = StringIO()
    writer = csv.writer(output)

    # Write header row
    writer.writerow(["Name", "Patrol", "Car Number", "Car Name", "Average Time", "Top Time", "Top Time Race #"])

    for p in participants:  # Assuming participants is your list of racers
        writer.writerow([p.first_name + " " + p.last_name, patrol_names.get(p.patrol), p.car_number, p.car_name, p.average_time, p.best_time, p.best_time_race_number])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=race_results.csv"},
    )


@app.route("/download_racer_data/<int:participant_id>")
def download_racer_data(participant_id):
    participant = next((p for p in participants if p.car_number == participant_id), None)
    if not participant:
        return "Participant not found", 404

    output = StringIO()
    writer = csv.writer(output)

    # Write header row
    writer.writerow(["Race #", "Time"])  # Customize headers as needed

    for i, time in enumerate(participant.times):
        race_number = next((race.race_number for race in races for heat in race.heats for lane, p in heat.lanes.items() if p == participant and i < len(heat.times) and heat.times[lane] == time), None)
        writer.writerow([race_number, time])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=racer_{participant_id}_data.csv"},
    )

@app.route("/upload_roster", methods=["GET", "POST"])
def upload_roster():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and (file.filename.endswith('.csv') or file.filename.endswith(".txt")): # Check file extension
            try:
                # Load from memory
                csv_file = file.stream.read().decode('utf-8')
                load_roster_from_memory(csv_file) # See function below
                save_data() # Save data after loading roster
                flash('Roster uploaded successfully!')
                return redirect(url_for('index')) # Redirect to your main page

            except Exception as e:
                flash(f'Error uploading roster: {str(e)}') # Display error message
                traceback.print_exc()

        else:
            flash('Invalid file type. Please upload a CSV file.')

        return redirect(request.url) # Redirect back to the upload page

    return render_template("upload_roster.html")

def load_roster_from_memory(csv_string):
    csvfile = StringIO(csv_string)
    reader = csv.DictReader(csvfile)
    for row in reader:
        p = add_participant(row["First Name"], row["Last Name"], row["Patrol"])
        if "car_weight_oz" in row:
            try:
                p.car_weight_oz = float(row["car_weight_oz"]) # Convert to float
            except ValueError:
                print(f"Invalid car_weight_oz: {row['car_weight_oz']}")  # Log the error

@app.route("/download_roster_template")
def download_roster_template():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["First Name", "Last Name", "Patrol", "car_weight_oz"]) # Header row
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=roster_template.csv"},
    )

@app.route("/api/participants")
def api_participants():
    return jsonify([p.toJSON() for p in participants])

@app.route("/api/patrol_names")
def api_patrol_names():
    return jsonify(patrol_names)

@app.route("/api/races")
def api_races():
    return jsonify([r.toJSON() for r in races])

@app.route("/api/designs")
def api_designs():
    return jsonify([d.toJSON() for d in designs])



if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
