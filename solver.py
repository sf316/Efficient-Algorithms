import os
import sys
sys.path.append('../../Desktop/170')
sys.path.append('../../Desktop')
import argparse
import utils

from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

INF = float('inf')


# Use Kruskal Algorithm to get MST
def get_mst(number_of_locations, key_locations, distance_matrix):
    number_of_key_locations = len(key_locations)

    union = []
    for i in range(number_of_locations):
        union += [i]

    def find_union(x):
        if union[x] == x: 
            return x
        union[x] = find_union(union[x])
        return union[x]

    def set_union(x, y):
        union_x = find_union(x)
        union_y = find_union(y)
        if union_x != union_y:
            union[union_x] = union_y

    all_edges = []
    for i in range(number_of_key_locations):
        for j in range(i + 1, number_of_key_locations):
            a = key_locations[i]
            b = key_locations[j]
            all_edges += [[a, b, distance_matrix[a][b]]]

    all_edges = sorted(all_edges, key = lambda x : x[2])

    mst_edges = []
    degrees = [0] * number_of_locations

    for edge in all_edges:
        if find_union( edge[0]) != find_union(edge[1]):
            mst_edges += [edge]
            set_union(edge[0], edge[1])

            degrees[edge[0]] += 1
            degrees[edge[1]] += 1

    return mst_edges, degrees


def get_minimum_matching(number_of_locations, distance_matrix, odd_degree_vertices):

    matched = [-1] * number_of_locations
    total_cost = 0
    for i in range(1, len(odd_degree_vertices), 2):
        a = odd_degree_vertices[i - 1]
        b = odd_degree_vertices[i]
        matched[a] = b
        matched[b] = a
        total_cost += distance_matrix[a][b]
    # print(f"total_cost = {total_cost}")

    augument_times = 10*1024 # can be changed here
    while augument_times > 0:
        augument_times -= 1

        augumented = 0
        for a in odd_degree_vertices:
            for c in odd_degree_vertices:
                if matched[a] != c:
                    b = matched[a]
                    d = matched[c]
                    dist_ab_cd = distance_matrix[a][b] + distance_matrix[c][d]
                    dist_ac_bd = distance_matrix[a][c] + distance_matrix[b][d]
                    dist_ad_bc = distance_matrix[a][d] + distance_matrix[b][c]
                    if dist_ac_bd < dist_ad_bc:
                        if dist_ab_cd > dist_ac_bd:
                            matched[a] = c
                            matched[c] = a
                            matched[b] = d
                            matched[d] = b
                            augumented = 1
                            total_cost += dist_ac_bd - dist_ab_cd
                            break
                        elif dist_ab_cd > dist_ad_bc:
                            matched[a] = d
                            matched[d] = a
                            matched[b] = c
                            matched[c] = b
                            augumented = 1
                            total_cost += dist_ad_bc - dist_ab_cd
                            break
            if augumented == 1:
                break

        # print(f"total_cost = {total_cost}")
        if augumented == 0:
            break;

    matched_edges = []
    for v in odd_degree_vertices:
        if v < matched[v]:
            matched_edges += [[v, matched[v], distance_matrix[v][matched[v]]]]

    return matched_edges



def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """

    number_of_locations = len(list_of_locations)
    name_to_index = {}
    home_list = []

    for i in range(len(list_of_locations)):
         name_to_index[list_of_locations[i]] = i

    start = name_to_index[starting_car_location]

    for home in list_of_homes:
        index = name_to_index[home]
        home_list += [index]

    home_set = set(home_list)

    # Floyd algorithm to calculate shortest paths between any 2 locations
    # (also for making the original graph a complete graph)
    dist = [[INF] * number_of_locations for _ in range(number_of_locations)]
    path = [[[ ]] * number_of_locations for _ in range(number_of_locations)]
    for i in range(number_of_locations):
        for j in range(number_of_locations):
            if adjacency_matrix[i][j] != 'x':
                dist[i][j] = adjacency_matrix[i][j]
                path[i][j] = [i, j]

    for k in range(number_of_locations):
        for i in range(number_of_locations):
            for j in range(number_of_locations):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    path[i][j] = path[i][k] + path[k][j][1:]


    def solve_with_key_locations(key_locations):
        # Kruskal algorithm to calculate minimum spanning tree T
        mst_edges, degrees = get_mst(number_of_locations, key_locations, dist)

        # Calculate the set of vertices O with odd degree in T
        odd_vertex_list = []
        for i in range(number_of_locations):
            if degrees[i] & 1:
                odd_vertex_list += [i]

        # Construct a minimum-weight perfect matching M in this subgraph
        matched_edges = get_minimum_matching(number_of_locations, dist, odd_vertex_list)

        # Unite matching M and spanning tree T to form an Eulerian multigraph
        edges_for_euler = matched_edges + mst_edges
        adj_for_euler = [[0] * number_of_locations for _ in range(number_of_locations)]
        for edge in edges_for_euler:
            a = min(edge[0], edge[1])
            b = max(edge[0], edge[1])
            adj_for_euler[a][b] += 1
            adj_for_euler[b][a] += 1

        # Hierholzer algorithm to calculate Euler tour
        def dfs(x, car_path):
            for y in range(number_of_locations):
                if x != y and adj_for_euler[x][y] > 0:
                    adj_for_euler[x][y] -= 1
                    adj_for_euler[y][x] -= 1
                    dfs(y, car_path)

            z = car_path[-1]
            car_path += path[z][x][1:]

        car_path = [start]
        dfs(start, car_path)

        total_cost = 0
        for i in range(1, len(car_path)):
            total_cost += dist[car_path[i - 1]][car_path[i]] * 2 / 3

        # drop TAs at nearest key locations
        nearest_location_on_path = {}
        car_path_set = set(car_path)
        for home in home_list:
            if home in car_path_set:
                nearest_location_on_path.update({home: home})
            else:
                nearest_location_on_path.update({home: start})
                dist_nearest = dist[start][home]
                for loc in key_locations:
                    if dist_nearest > dist[loc][home]:
                        dist_nearest = dist[loc][home]
                        nearest_location_on_path[home] = loc

        drop_offs = {}
        for home, drop_off in nearest_location_on_path.items():
            total_cost += dist[drop_off][home] if drop_off != home else 0

            if drop_off in drop_offs:
                drop_offs[drop_off] += [home]
            else:
                drop_offs.update({drop_off: [home]})

        return total_cost, car_path, drop_offs


    # Set of all key vertices (starting vectex and TA homes)
    key_location_set = set(home_list + [start])
    
    # Get solution with original key locations
    total_cost, car_path, drop_offs = solve_with_key_locations(list(key_location_set))

    # Try to remove some key locations
    optimization_times = min(len(home_list), 200)
    while optimization_times > 0:
        optimization_times -= 1

        optimized = 0
        for loc in key_location_set:
            if loc != start:
                new_key_location_set = key_location_set - set([loc])
                new_total_cost, new_car_path, new_drop_offs = solve_with_key_locations(list(new_key_location_set))

                if total_cost > new_total_cost:
                    total_cost, car_path, drop_offs = new_total_cost, new_car_path, new_drop_offs
                    next_key_location_set = new_key_location_set
                    optimized += 1

        if optimized == 0:
            break 

        key_location_set = next_key_location_set

    # print(f"total_cost = {total_cost}")
    return car_path, drop_offs


"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
