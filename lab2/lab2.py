import numpy as np

def check_initial_data(supply : list, demand : list, costs : np.ndarray):
    if sum(supply) < sum(demand):
        supply.append(sum(demand) - sum(supply))
        row_to_be_added = np.array([[0 for _ in range(len(demand))]])
        costs = np.append(costs, row_to_be_added, axis=0)
        return supply, demand, costs
    
    if sum(demand) < sum(supply):
        demand.append(sum(supply) - sum(supply))
        column_to_be_added = np.array([[0 for _ in range(len(supply))]])
        costs = np.append(costs, column_to_be_added, axis=1)
        return supply, demand, costs

    return supply, demand, costs


def create_reference_plan(supply : list, demand : list):
    s = supply.copy()
    d = demand.copy()

    plan = np.full((len(supply), len(demand)), -1)

    i, j = 0, 0

    degenerate_index = ()

    while i < len(supply) and j < len(demand):
        x = min(s[i], d[j])
        plan[i][j] = x

        s[i] -= x
        d[j] -= x

        if s[i] == 0 and d[j] == 0 and (i == j and i != len(supply) - 1 and j != len(supply) - 1):
            degenerate_index = (i, j)
            plan[i + 1][j] = 0
            i += 1
            j += 1
        
        elif s[i] == 0:
            i += 1
        
        elif d[j] == 0:
            j += 1
        
    print(plan)
    return plan, degenerate_index


def calculate_potentials(plan : np.ndarray, costs : np.ndarray, supply_len : int, demand_len : int):
    u = [None] * supply_len
    v = [None] * demand_len
    
    u[0] = 0

    while None in u or None in v:
        for i in range(supply_len):
            for j in range(demand_len):
                if plan[i][j] > -1:
                    if u[i] is not None and v[j] is None:
                        v[j] = costs[i][j] - u[i]    
                    elif v[j] is not None and u[i] is None:
                        u[i] = costs[i][j] - v[j]
    return u, v


def find_cycle(plan, i0, j0):
    rows, cols = plan.shape
    used = np.zeros_like(plan, dtype=bool)

    path = []

    def dfs(i, j, row_move=True):
        if used[i][j]:
            return path if (i, j) == (i0, j0) and len(path) >= 4 else None
        
        used[i][j] = True
        path.append((int(i), int(j)))

        if row_move:
            for k in range(cols):
                if plan[i][k] > -1 or (i == i0 and k == j0):
                    result = dfs(i, k, row_move=False)
                    if result: 
                        return result
        else:
            for k in range(rows):
                if plan[k][j] > -1 or (k == i0 and j == j0):
                    result = dfs(k, j, row_move=True)
                    if result:
                        return result
        path.pop()
        return None
    
    return dfs(i0, j0)


def calculate_total_cost(plan : np.ndarray, costs : np.ndarray, supply_len : int, demand_len : int):
    total_cost = 0
    for i in range(supply_len):
        for j in range(demand_len):
            if plan[i][j] == -1:
                continue
            total_cost += plan[i][j] * costs[i][j]
    
    return total_cost


def improve_plan(plan : np.ndarray, costs : np.ndarray, supply_len : int, demand_len : int, degenerate_index : tuple):
    iteration = 0
    is_error = False
    while True:
        iteration += 1
        print(f"\n=== Plan №{iteration} ===")

        u, v = calculate_potentials(plan, costs, supply_len, demand_len)
        print(f"Potentials: u: {u}, v: {v}")

        deltas = np.full((supply_len, demand_len), np.nan)

        for i in range(supply_len):
            for j in range(demand_len):
                if plan[i][j] == -1:
                    deltas[i][j] = (u[i] + v[j]) - costs[i][j]
        
        print(f"Deltas:\n{deltas}")

        if np.nanmax(deltas) <= 0:
            print("The solution is optimized.")
            return plan
        
        i0, j0 = np.unravel_index(np.nanargmax(deltas), deltas.shape)
        print(f"Need to improve in position: ({int(i0)}, {int(j0)})")

        cycle = find_cycle(plan, i0, j0)
        k = 0
        while k < 2:
            if not cycle or len(cycle) < 4:
                if k == 0:
                    plan = plan[degenerate_index[i]][degenerate_index[j] + 1]
                    find_cycle(plan, i0, j0)
                
                if k == 1:
                    plan = plan[degenerate_index[i]][degenerate_index[j] - 1]
                    find_cycle(plan, i0, j0)
            else:
                break

            k += 1
        
        if k >= 2:
            print("Cycle not found or impossible") 
            is_error = True
            break

        cycle = [(int(i), int(j)) for i, j in cycle]
        print(f"Cycle is found: {cycle}")

        cycle_values = [(plan[i][j], (i, j)) for i, j in cycle[1::2]]
        if not cycle_values:
            print("Failed to find valid cycle moves.")
            is_error = False
            break

        theta_tuple = min(cycle_values)
        theta = theta_tuple[0]
        print(f"Minimum value of cycle: {theta}")
        
        for k, (i, j) in enumerate(cycle):
            if k % 2 == 0:
                plan[i][j] += theta + int(not bool(plan[i][j] + 1))
            else:
                plan[i][j] -= theta + int((i, j) == theta_tuple[1])
        
        print(f"Plan value:\n{plan}")
        interim_cost = calculate_total_cost(plan, costs, supply_len, demand_len)
        print(f"Interim total cost of transportation: {interim_cost}")
    
    if is_error:
        return False


def main():
    supply = [45, 35, 70]
    demand = [20, 60, 55, 45]
    costs = np.array([
        [2, 5, 3, 4],
        [6, 1, 2, 5],
        [3, 4, 3, 8],
    ])

    supply, demand, costs = check_initial_data(supply, demand, costs)
    plan, degenerate_index = create_reference_plan(supply, demand)
    temp = improve_plan(plan, costs, len(supply), len(demand), degenerate_index)

    if type(temp) == bool:
        raise Exception("An error occurred during plan improvement.")
    
    print(f"Plan value:\n{plan}")
    print(f"Final transportation cost: {calculate_total_cost(plan, costs, len(supply), len(demand))}")


if __name__ == "__main__":
    main()
