from flask import Flask, render_template, request, redirect, url_for
import itertools
import json

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
                assign_car_numbers()
                return redirect(url_for("index"))
            else:
                return render_template("index.html", participants=participants, error=f"Subgroup must be between 1 and {len(subgroup_names)}.")
        except ValueError:
            return render_template("index.html", participants=participants, error="Invalid subgroup input.")
    return render_template("index.html", participants=participants, subgroup_names=subgroup_names)

def add_participant(name, subgroup):
    participants.append(Participant(name, subgroup))

def assign_car_numbers():
    for subgroup in subgroup_names:
        car_number = 1
        for p in participants:
            if p.subgroup == subgroup:
                p.car_number = car_number
                p.car_name = f"{subgroup_names[subgroup][0]}{car_number}"
                car_number += 1

@app.route("/schedule")
def schedule():
    schedule_races()
    return render_template("schedule.html", races=races, subgroup_names=subgroup_names)

def schedule_races():
    races.clear()
    heat_number = 1

    for subgroup in subgroup_names:
        subgroup_participants = [p for p in participants if p.subgroup == subgroup]
        num_participants = len(subgroup_participants)

        if num_participants >= 3:
            for round_num in [1, 2] if subgroup != "Exhibition" else [1]:
                if round_num == 2:
                    subgroup_participants = sorted(subgroup_participants, key=lambda p: (p.average_time, p.best_time))[:4]
                    num_participants = len(subgroup_participants)

                # Create combinations and assign lanes:
                for i in range(num_participants):
                    p1 = subgroup_participants[i]
                    for j in range(i + 1, num_participants):
                        p2 = subgroup_participants[j]
                        for k in range(j + 1, num_participants):
                            p3 = subgroup_participants[k]
                            for l in range(k + 1, num_participants + 1):
                                p4 = None if l > num_participants else subgroup_participants[l-1]

                                # Check if participants have raced twice already
                                if p1.times and len(p1.times) >= 2: continue
                                if p2.times and len(p2.times) >= 2: continue
                                if p3.times and len(p3.times) >= 2: continue
                                if p4 and p4.times and len(p4.times) >= 2: continue

                                # Ensure outer/inner lane balance (simplified):
                                # Create two heats, swapping lanes
                                for heat_num in [1, 2]:
                                    race = Race(heat_number)
                                    if heat_num == 1:
                                        race.lanes[1] = p1
                                        race.lanes[2] = p2
                                        race.lanes[3] = p3
                                        if p4: race.lanes[4] = p4
                                    else:
                                        race.lanes[1] = p2
                                        race.lanes[2] = p1
                                        race.lanes[3] = p4 if p4 else p3
                                        race.lanes[4] = p3 if p4 else None

                                    races.append(race)
                                    heat_number += 1
                                break
                        break
                    break

    # Final Round
    fastest_racers = []
    for subgroup in subgroup_names:
        subgroup_participants = [p for p in participants if p.subgroup == subgroup and subgroup != "Exhibition"]
        if subgroup_participants:
             fastest = sorted(subgroup_participants, key=lambda p: (p.average_time, p.best_time))[0]
             fastest_racers.append(fastest)

    if len(fastest_racers) > 1:
        final_race = Race(heat_number)
        for i, racer in enumerate(fastest_racers):
            final_race.lanes[i+1] = racer
        races.append(final_race)


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
