from ast import Num
from flask import Flask, render_template, request, redirect, url_for, \
    jsonify, Response, flash, session
import itertools
import json
import os
import os.path
from enum import Enum
import csv
from io import StringIO
import traceback
import uuid
import secrets
import qrcode


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

try:
    with open("auth_config.json", 'r') as f:
        auth_config = json.load(f)
except FileNotFoundError:
    auth_config = {}

NUM_LANES = 4

DATA_FILE = "race_data.json"  # Define the filename for storing data

class Role(Enum):
    PUBLIC = 0
    JUDGE = 1
    OWNER = 2

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
designs = {}
judging_active = True
judges = {}

def load_data():
    resave = False
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            # Reconstruct participants, races, etc. from loaded data
            global participants, races, initial_races_completed, semi_final_races_completed
            global designs
            global judges
            global judging_active

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

            designs = {}
            for p_id,d_data in data.get("designs", {}).items():
                participant = next((p for p in participants if p.participant_id == p_id), None)
                if participant:
                    d = Design(participant)
                    d.scores = d_data.get("scores", {}) # Load scores
                    designs[p_id] = d
            for p in participants:
                if p.participant_id not in designs:
                    resave = True
                    designs[p.participant_id] = Design(p)

            judges = data.get("judges", {})
            initial_races_completed = data.get("initial_races_completed", {})
            semi_final_races_completed = data.get("semi_final_races_completed", {})
            judging_active = data.get("judging_active", True)

    except FileNotFoundError:
        save_data()

    if resave:
        save_data()

def save_data():
    data = {
        "participants": [p.toJSON() for p in participants],
        "races": [r.toJSON() for r in races],
        "designs": {p_id: d.toJSON() for p_id, d in designs.items()},
        "judges": judges,
        "judging_active": judging_active,
        "initial_races_completed": initial_races_completed,
        "semi_final_races_completed": semi_final_races_completed,
    }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, default=str)  # Use default=str to handle non-serializable objects like Enums

# Load data when the app starts
try:
    load_data()
except Exception as e:
    print(f"Encountered exception loading saved data")
    traceback.print_exc()

@app.route("/owner_login", methods=["POST"])
def owner_login():
    token = request.form.get("token")
    if token == auth_config.get("owner_token"):
        session["role"] = Role.OWNER.name
        return redirect(url_for("index"))  # Redirect to owner's page
    return render_template("login.html", error_message="Invalid token"), 401

def generate_qr(url,id):

    qr = qrcode.QRCode(box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    image = qr.make_image()

    filename = f"{id}.png"
    file_path=os.path.join(os.curdir, "static", "qr", filename)
    image.save(file_path)

    return filename

@app.route("/judge_login", methods=["POST"])
def judge_login():
    global judges
    role = session.get('role')
    token = request.form.get("token")
    judge_name = request.form.get("judge_name", None)
    if token == auth_config.get("judge_token"):
        if judge_name not in judges:
            if role and Role[role] == Role.OWNER:
                id = uuid.uuid4().hex
                judges[judge_name] = {}
                judges[judge_name]['id'] = id
                url = request.url_root.rstrip('/') + url_for("login", judge_token=token, judge_name=judge_name)
                judges[judge_name]["qr"] = generate_qr(url, id )
                save_data()
            else:
                return render_template("login.html", error_message="Only the owner can create new Judge logins"), 403

        # Owner can create judge roles, but isn't a judge
        if (role and Role[role] != Role.OWNER) or not role:
            session["role"] = Role.JUDGE.name
            session["judge_name"] = judge_name
            return redirect(url_for("judge_design"))  # Redirect to judge's page
        else:
            return redirect(url_for("login", judge_name=judge_name)) # Owner might be making multiple judges
    return render_template("login.html", error_message="Invalid token"), 401

@app.route("/logout")
def logout():
    session.pop("role", None)  # Remove the role from the session
    session.pop("judge_name", None)
    return redirect(url_for("index"))  # Or wherever you want to redirect

@app.route("/login")
def login():
    role = session.get('role')
    token = request.args.get("judge_token", None)
    judge_name = request.args.get("judge_name", None)
    if role and Role[role] == Role.OWNER and judge_name in judges:
        judge_token = auth_config.get('judge_token')
        judge_qr = url_for('static', filename=os.path.join('qr', judges[judge_name]['qr']))
    else:
        judge_qr = None
        judge_token = token
    if judge_name is not None:
        return render_template("login.html", judge_name=judge_name, judge_token=judge_token, judge_qr=judge_qr)
    else:
        return render_template("login.html")


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

@app.route("/edit_participant/<participant_id>", methods=["GET", "POST"])
def edit_participant(participant_id):
    participant = next((p for p in participants if p.participant_id == participant_id), None) 
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

@app.route("/delete_participant/<participant_id>", methods=["POST"])
def delete_participant(participant_id):
    participant = next((p for p in participants if p.participant_id == participant_id), None)
    if not participant:
        return "Participant not found", 404

    if participant_id in designs:
        del designs[participant_id]
    participants.remove(participant)  # Remove the participant

    save_data() # Save data after removing participant
    return redirect(url_for("index"))  # Redirect back to the participant list

@app.route("/participant_times/<participant_id>")
def participant_times(participant_id):
    participant = next((p for p in participants if p.participant_id == participant_id), None)
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
        new_p.car_name = f"{patrol_names.get(patrol)[:1]}{next_car_number:02}"

        participants.append(new_p)

        designs[new_p.participant_id] = Design(p)

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
    name_sorted_top_racers = sorted(top_racers, key=lambda racer: racer.car_name)

    if top_racers:
        race_groups = [name_sorted_top_racers]  # Create a single group of top racers
        assign_paired_lanes(race_groups, Rounds.SEMI, len(races) + 1) 

def schedule_final_races():
    global races
    if not all(semi_final_races_completed.values()):
        return

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
        filtered_races = [r for r in races if r.round == round and r.patrol != "Exhibition"]
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
    role = session.get('role', Role.PUBLIC.name)
    error_message = None
    race = next((r for r in races if r.race_number == race_number), None)
    if not race:
        return "Race not found", 404
    heat = next((h for h in race.heats if h.heat_number == heat_number), None)
    if not heat:
        return "Heat not found", 404

    if Role[role] != Role.OWNER:
        error_message = "This page is only intended for the Owner, please log in."
    else:

        if request.method == "POST":
            submit = request.form.get("submit", None)
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

            if heat_number == len(race.heats):
                next_race = next((r for r in races if r.race_number == race_number+1), None)
                if next_race is not None and next_race.patrol == race.patrol:
                    next_race_number = next_race.race_number
                    next_heat_number = 1
                else:
                    next_race_number = None
                    next_heat_number = None
            else:
                next_race_number = race.race_number
                next_heat_number = heat_number + 1
                if next((r for r in races if r.race_number == next_race_number), None) == None:
                    next_race_number = None
                    next_heat_number = None
                    
            if submit == "Submit & Go to Next Race" and next_race_number is not None:
                # Go to the next race
                return redirect(url_for("enter_times", race_number=next_race_number, heat_number=next_heat_number)) # Add round parameter
            else:
                # Go to the list
                return redirect(url_for("schedule", patrol=race.patrol, round=race.round.value)) # Add round parameter

    return render_template("enter_times.html", error_message=error_message, race=race, heat=heat, NUM_LANES=NUM_LANES)

def check_races_complete(patrol, round):
    filtered_races = [r for r in races if r.patrol == patrol and r.round == round]
    filtered_heats = [h for r in filtered_races for h in r.heats]
    total_assigned_lanes = len([l for h in filtered_heats for l in h.lanes if h.lanes[l] is not None])
    total_times = len([t for h in filtered_heats for t in h.times if h.times[t] is not None])

    return total_assigned_lanes == total_times if total_assigned_lanes > 0 else None

def check_races_scheduled(patrol, round):
    filtered_races = [r for r in races if r.patrol == patrol and r.round == round]
    filtered_heats = [h for r in filtered_races for h in r.heats]
    total_assigned_lanes = len([l for h in filtered_heats for l in h.lanes if h.lanes[l] is not None])

    return total_assigned_lanes > 0

def check_round_complete(round):
    global initial_races_completed, semi_final_races_completed
    for p in patrol_names:
        if p == "Exhibition":
            semi_final_races_completed[p] = True
        else:
            complete = check_races_complete(p, round)
            if complete is not None:
                if round == Rounds.FIRST:
                    initial_races_completed[p] = complete
                elif round == Rounds.SEMI:
                    semi_final_races_completed[p] = complete

@app.route("/schedule_initial")
def schedule_initial():
    clear_races()
    schedule_initial_races()
    save_data() # Saving data after scheduling initial round
    return redirect(url_for("schedule", round=Rounds.FIRST.value)) # Redirect to the main schedule page

@app.route("/schedule_semifinal/<patrol>")
def schedule_semifinal(patrol):
    schedule_semi_final_races(patrol)
    save_data() # Saving data after scheduling semi-final round
    return redirect(url_for("schedule", round=Rounds.SEMI.value)) # Add round parameter

@app.route("/schedule_final")
def schedule_final():
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
    if selected_round == Rounds.FINAL:
        selected_patrol = ""

    selected_round_name = ""
    if selected_round == Rounds.FIRST:
        selected_round_name = "First Round"
    elif selected_round == Rounds.SEMI:
        selected_round_name = "Semi-Finals"
    elif selected_round == Rounds.FINAL:
        selected_round_name = "Finals"

    check_round_complete(Rounds.FIRST)
    check_round_complete(Rounds.SEMI)

    top_racers, overall_racer_averages = get_top_racers(selected_round, selected_patrol, NUM_LANES)

    if semi_final_races_completed:
        all_semi_final_races_completed = all([v for k,v in semi_final_races_completed.items() if k != "Exhibition"])
    else:
        all_semi_final_races_completed = False
    semi_final_races_scheduled = {p: check_races_scheduled(p, Rounds.SEMI) for p in patrol_names}

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
                           all_semi_final_races_completed=all_semi_final_races_completed,
                           semi_final_races_scheduled=semi_final_races_scheduled )

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

    role = session.get('role', Role.PUBLIC.name)
    judge_name = session.get('judge_name', None)
    judge_id = judges[judge_name]['id'] if judge_name else None
    error_message = ""

    racers = {}
    for patrol in patrol_names:
        racers[patrol] = [p for p in participants if p.patrol == patrol]

    if not judging_active:
        error_message = "Judging is closed."

    if not error_message and Role[role] != Role.JUDGE:
        error_message = "This page is only for authorized Judges, please log in."

    if not error_message and request.method == "POST":

        for p in participants:
            rank = request.form.get(f"rank_{p.participant_id}", 0)
            designs[p.participant_id].scores[judge_id] = int(rank)
        save_data()
        return redirect(url_for("design_results"))

    return render_template("judge_design.html", racers=racers, patrol_names=patrol_names,
                            judge_id=judge_id, designs=designs, judge_name=judge_name, 
                            judging_active=judging_active, error_message=error_message)

@app.route("/close_judging")
def close_judging():
    global judging_active
    judging_active = False
    save_data()
    return redirect(url_for("design_results"))

@app.route("/open_judging")
def open_judging():
    global judging_active
    judging_active = True
    save_data()
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
                design_score = score_design(designs[p.participant_id])
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


@app.route("/download_racer_data/<participant_id>")
def download_racer_data(participant_id):
    participant = next((p for p in participants if p.participant_id == participant_id), None)
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
    role = session.get('role', Role.PUBLIC.name)
    error_message = ""

    if Role[role] == Role.OWNER:
        if request.method == "POST":
            if 'file' not in request.files:
                flash('Must provide file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('Must provide file')
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
    else:
        error_message="Only the owner can upload racer data."

    return render_template("upload_roster.html", error_message="")

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
    for p in participants:
        writer.writerow([p.first_name, p.last_name, patrol_names[p.patrol], p.car_weight_oz])
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
    return jsonify({p: d.toJSON() for p, d in designs.items()})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
