import math
import random
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import pandas as pd
import plotly.express as px

#user inputs

Lambda = 2.25
mew = 8.98
st1 = 7 #lb for gg service
st2 = 5 #ub for gg service
#mean for gg ia normal
sigma = 1.5 #sigma for gg ia normal
c = 2


A = 55
M = 1994
Z0 = 10112166
C = 9
a = 1
b = 3

def CP(Lambda):
    array = []
    i=1
    total = 0
    num_of_cust = 0
    while total != 1:
        total = 0 
        for x in range (0, i):
            temp = (Lambda**x) * math.exp(-Lambda) / math.factorial(x)
            total += temp
        array.append(total)
        i+=1
        num_of_cust+=1
    return array, num_of_cust

def CPlookUp(Lambda, num_of_cust):
    array = []
    for i in range (0, num_of_cust):
        total = 0
        for x in range (0, i):
            temp = (Lambda**x) * math.exp(-Lambda) / math.factorial(x)
            total += temp
        array.append(total)
    return array

def IAMM(CP, CPlo, num_of_cust):
    IA = []
    for j in range (1, num_of_cust):
        temp = random.random()
        for i in range (0, num_of_cust - 1):
            if (temp<CP[i] and temp>CPlo[i]):
                IA.append(i)
    return IA

def IAMG(CP, CPlo, num_of_cust):
    IA = []
    while (len(IA)!=num_of_cust):
        temp = -mew * math.log(random.random())
        for i in range (0, num_of_cust - 1):
            if (temp<CP[i] and temp>CPlo[i]):
                IA.append(i)
    return IA

def IAGG(CP, CPlo, num_of_cust, mew):
    IA = []
    while (len(IA)!=num_of_cust):
        #temp = np.random.normal(mew, sigma)
        temp = -mew * math.log(random.random())
        for i in range (0, num_of_cust - 1):
            if (temp<CP[i] and temp>CPlo[i]):
                IA.append(i)
    return IA

def Arrivals(arrivals, IA, num_of_cust):
    temp = 0
    print(len(IA), num_of_cust)
    for i in range (0, num_of_cust - 1):
        temp += IA[i]
        arrivals.append(temp)

def ServiceMM(num_of_cust, mew):
    service = []
    for i in range (0, num_of_cust):
        temp = -mew * math.log(random.random())
        service.append(round(temp))
    return service

def ServiceMG(num_of_cust, mew, sigma):
    service = []
    for i in range (0, num_of_cust):
        temp = np.random.normal(mew, sigma)
        service.append(round(temp))
    return service

def ServiceGG(num_of_cust, st1, st2):
    service = []
    for i in range (0, num_of_cust):
        temp = (random.random()-st1)/(st2-st1)
        service.append(round(temp))
    return service

def generate_priority(A, M, Z0, C, a, b, num_of_cust):
    Z = [Z0]
    R = []
    RanNum = []
    GP =[]
    for i in range (0, num_of_cust):
        temp = (A*(Z[i])+C) % M
        Z.append(temp)
        R.append(Z[i+1])
        RanNum.append(R[i]/M)
        priority = a + RanNum[i] * (b - a)
        GP.append(round(priority))
    Z.remove(Z[-1])
    return Z, R, RanNum, GP


def qeueing(num_of_cust, arrivals, service):
  
  Z, R, RanNum, GP = generate_priority(A, M, Z0, C, a, b, num_of_cust)
  arrived = []
  labels = []
  for i in range(len(arrivals)):
        arrived.append({
        "name": f"Patient {i + 1}",
        "arrival_time": arrivals[i],
        "service_time": service[i],
        "priority": GP[i],
        })
        labels.append(f"Name {i+1}")

  starts = []   
  width = []   
  table = PrettyTable([
    "Name", "Arrival Time", "Service Time", "Priority", "Service Start Time",
    "Service End Time", "Turnaround Time", "Wait Time", "Response Time"
        ])
  time = 0          #simulation clock intialised
  n = len(arrived)
  executed = 0
  current = 0
  waiting_queue = []
  executed_processes = set()
  remaining_times = {
      p["name"]: [p["service_time"], p["priority"], None, 0]
      for p in arrived
  }
  total_service_time = 0
  total_busy_time = 0

  while executed < n:
    for p in arrived:
      if p["arrival_time"] == time and p["name"] not in executed_processes:
        waiting_queue.append(p)

        if (not current) or (waiting_queue and remaining_times[p["name"]][1] < remaining_times[current["name"]][1]):
          if current:
            print(
                "\n"
                f"Leaving {current['name']} and Switching to process {p['name']} due to priority."
                "\n")
          current = p

    if not current and waiting_queue:
      current = min(waiting_queue, key=lambda x: remaining_times[x["name"]][1])

    if not current and not waiting_queue:
      print("\n"
            f"Time {time}: Server is idle."
            "\n")

    if current:
      print(f"Time {time}: Executing {current['name']}")

      if remaining_times[current["name"]][2] == None:
        remaining_times[current["name"]][2] = time  # Service start time

      if remaining_times[current["name"]][0] > 0:
        remaining_times[current["name"]][0] -= 1
        total_service_time += 1
        total_busy_time += 1

      if remaining_times[current["name"]][0] <= 0:
        executed_processes.add(current["name"])
        remaining_times[current["name"]][3] = time # Service end time
        waiting_queue = [
            p for p in waiting_queue if p["name"] != current["name"]
        ]

        turnaround_time = remaining_times[
            current["name"]][3] - current["arrival_time"]

        wait_time = max(turnaround_time - current["service_time"], 0)

        response_time = remaining_times[
            current["name"]][2] - current["arrival_time"]

        table.add_row([
            current["name"], current["arrival_time"], current["service_time"],
            current["priority"], remaining_times[current["name"]][2],                   #table index 4 and 5 have start and end time respectively
            remaining_times[current["name"]][3], turnaround_time, wait_time,
            response_time
        ])
        width.append(remaining_times[current["name"]][3] - remaining_times[current["name"]][2])
        starts.append(remaining_times[current["name"]][2])
        current = None
        executed += 1
        
    # df = pd.DataFrame({
    #     'Task': labels,
    #     'Start': starts,
    #     'Finish': [starts[i] + width[i] for i in range(len(starts))],
    #     'Priority': [arrived[i]['priority'] for i in range(len(arrived))]
    # })

    time += 1
  server_utilization_rate = total_busy_time / time
#   plot_gantt_chart(starts, width, labels)


  print(table)
  #plt.barh(range(len(labels)), width, left=starts, tick_label=labels)
  #print("\n")
  #print(f"\nTotal Service Time: {total_service_time}")

  total_waiting_time = 0
  total_turnaround_time = 0
  total_response_time = 0

  for p in arrived:
    if p["name"] in executed_processes:
      turnaround_time = remaining_times[p["name"]][3] - p["arrival_time"]
      wait_time = max(turnaround_time - p["service_time"], 0)
      response_time = remaining_times[p["name"]][2] - p["arrival_time"]

      total_waiting_time += wait_time
      total_turnaround_time += turnaround_time
      total_response_time += response_time

  # Calculate averages
  average_waiting_time = total_waiting_time / n
  average_turnaround_time = total_turnaround_time / n
  average_response_time = total_response_time / n

  # Print the results
  print("\n")
  print(f"Average Waiting Time: {average_waiting_time:.2f}")
  print(f"Average Turnaround Time: {average_turnaround_time:.2f}")
  print(f"Average Response Time: {average_response_time:.2f}")
  print(f"\nServer Utilization Rate: {server_utilization_rate * 100:.2f} %")

def plot_gantt_chart(starts, width, labels, df):
    fig = px.timeline(
        df, x_start='Start', x_end='Finish', y='Task', color='Priority',
        labels={"Task": "Name"}
    )
    fig.update_yaxes(categoryorder='total ascending')  # Order tasks by Name
    fig.update_layout(
        title='Gantt Chart for Queueing Simulation',
        xaxis_title='Simulation Time',
        yaxis_title='Task',
        showlegend=False  # Hide legend for better readability
    )
    fig.show()


def calculate_p0(rho, c):
    numerator = (rho ** c) / math.factorial(c)
    denominator = sum([(rho ** k) / math.factorial(k) for k in range(c)])
    pzero = 1 / (numerator + denominator)
    return pzero


def mmc(Lambda, mew, c):
    arrival_rate = Lambda
    service_rate = mew
    
    utilization = arrival_rate / (service_rate*c)
    p0 = calculate_p0(utilization, c)
    lq = p0*((Lambda/mew)**c)*utilization
    wq = lq / arrival_rate
    w = wq + (1/ service_rate)
    l = arrival_rate * w
    
    results = {
        'Utilization': utilization,
        'Probability of zero customers': p0,
        'Average number of customers in the system': l,
        'Average number of customers in the queue': lq,
        'Average waiting time in the system': w,
        'Average waiting time in the queue': wq
        }
    print(results)

def mgc(Lambda, mew, c):
    
    arrival_rate = Lambda
    service_rate = mew
    mean_arrival = 1/Lambda
    mean_service = 1/mew
    var_arrival = 1/Lambda**2
    var_service = 1/mew**2

    Ca2= var_arrival/((mean_arrival)**2)
    Cs2= var_service/((mean_service)**2)
    
    utilization = arrival_rate / (service_rate*c)
    p0 = calculate_p0(utilization, c)
    lq = p0*((Lambda/mew)**c)*utilization
    approx = (Ca2+Cs2)/2
    wq = (lq / arrival_rate)*(approx)
    w = wq + (1/ service_rate)
    l = arrival_rate * w
    
    results = {
        'Utilization': utilization,
        'Probability of zero customers': p0,
        'Average number of customers in the system': l,
        'Average number of customers in the queue': lq,
        'Average waiting time in the system': w,
        'Average waiting time in the queue': wq
        }
    print(results)

def ggc(Lambda, st1, st2, c):
    arrival_rate = Lambda
    service_rate = mew

    mean_arrival = 1/Lambda
    mean_service = 1/((st1+st2)/2)
    var_arrival = (1/Lambda)**2
    var_service= ((st2 - st1)**2) / 12

    Ca2= var_arrival/(mean_arrival**2)
    Cs2= var_service/(mean_service**2)

    utilization = service_rate / (arrival_rate*c)
    p0 = calculate_p0(utilization, c)
    lq = p0*((Lambda/mew)**c)*utilization
    approx = (Ca2+Cs2)/2
    wq = (lq / arrival_rate)*(approx)
    w = wq + (1/ service_rate)
    l = arrival_rate * w

    results= {
        'Utilization': utilization,
        'Probability of zero customers': p0,
        'Average number of customers in the system': l,
        'Average number of customers in the queue': lq,
        'Average waiting time in the system': w,
        'Average waiting time in the queue': wq
    }
    print(results)
    

def main():

    arr1, num_of_cust = CP(Lambda)
    arr2 = CPlookUp(Lambda, num_of_cust)
    #IA = IAMM(arr1, arr2, num_of_cust)
    #IA = IAMG(arr1, arr2, num_of_cust)
    #print(IA)
    IA = IAGG(arr1, arr2, num_of_cust, mew)
    arrivals = [0]
    Arrivals(arrivals, IA, num_of_cust)
    #print(Arrivals)
    IA.insert(0, 0)
    #service = ServiceMM(num_of_cust, mew)
    #service = ServiceMG(num_of_cust, mew, sigma)
    service = ServiceGG(num_of_cust, st1, st2)
    qeueing(num_of_cust, arrivals, service)

    #Possible prompts:

    #mmc(Lambda, mew, c)
    #mgc(Lambda, mew, c)
    ggc(Lambda, st1, st2, c)
    #plt.show()


if __name__ == "__main__":
    main()