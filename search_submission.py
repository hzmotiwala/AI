# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

import heapq
import os
import pickle
import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        
        
      
        #print ("initial frontier: ", PriorityQueue())
        #heap = self.queue
        #print ("heap: ", heap)
        #heap_after_pop = heapq.heappop(self.queue)
        #print ("heap after pop: ", heap_after_pop)
        return heapq.heappop(self.queue)
        #raise NotImplementedError


    def remove(self, node_id):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
        debug = False

        if debug:
            print("queue: ", self.queue)
            print("node id: ", node_id)

        for item in self.queue:
            print(item)



        heap = self.queue
        ordered = list(heap)
        ordered.pop(node_id)
        #print("ordered: list" , ordered)
        #ordered = heapq.heapify(ordered)

        #heapq.heappop((heap)[2])
        #heap = self.queue
        #print ("ordered: ", ordered)
        #print ("ordered type: ", type(ordered))

        queue = PriorityQueue()
        #print("queue 1: ", queue)
        for item in ordered:
            queue.append(item)
        #print("queue 2: ", queue)

        #self.queue = queue

        #print ("queue: ", queue)
        #print ("queue type: ", type(queue))

        

        return queue

    def remove_if_same_path(self, path):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
        debug = False

        if debug:
            print("")
            print(" REMOVE IF SAME PATH START *** \n")
            print("current queue: ", self.queue)
            print("path to look for removal: ", path)

        queue = PriorityQueue()

        for item in self.queue:
            this_node, this_path = item
            if debug:
                print("")
                print("item: ", item)
                print("this_node: ", this_node)
                print("this_path: ", this_path)
            if path != this_path:
                queue.append(item)

        if debug:
            print("queue after removal: ", queue)
            print("\n")

        return queue


    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """


        #heap = self.queue
        #print ("heap: ", heap)
        heapq.heappush(self.queue,node)
        #heap = self.queue
        #print ("heap after append: ", heapq.heappush(heap,node))
        #print ("new frontier after append: ", PriorityQueue(self.queue))

        #heapq.heappush(self.queue, node)
        #raise NotImplementedError

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n for _, n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """



    debug = False
    show_result = False

    if debug:
        print ("start: ", start)
        print ("goal: ", goal)

    #based on book page 82
    if start == goal:
        return []

    frontier = PriorityQueue()
    frontier.append((0, [start]))

    if debug:
        print("first frontier: ", frontier)

    explored_nodes = []

    while frontier:
        if debug:
            print("\n frontier: ", frontier)
        node, path = frontier.pop()
        this_node = path[-1]
        if debug:
            print("\nnew node: ", node)
            print("new path: ", path)
            print("this node: ", this_node)
        if this_node not in explored_nodes:
            for neighbor in sorted(graph.neighbors(this_node), key=str.lower):
            #for neighbor in graph.neighbors(this_node):
                if debug:
                    print("neighbor sorted: ", neighbor)
                if neighbor not in explored_nodes:
                    new_path = list(path)
                    new_path.append(neighbor)
                    length = len(new_path)
                    if debug:
                        print("new path to be added", new_path)
                    frontier.append((length, new_path))
                    if neighbor == goal:
                        if show_result:
                            print ("goal return: ", new_path)
                        return new_path
            explored_nodes.append(this_node)
            if debug:
                print("explored nodes: ", explored_nodes)
    return []
    
    

def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """


    debug = False
    show_result = False

    if start == goal:
        return []
    
    frontier = PriorityQueue()
    frontier.append((0, [start]))



    #frontier.append((2, ['a', 'b', 'c']))
    #frontier.append((1, ['c', 'f', 'd']))
    #frontier.append((17, ['c', 'f', 't']))
    #frontier.append((4, ['x', 'j', 'd']))

    #frontier.remove(4)


    explored_nodes = []    

    while frontier: # and top(frontier) is cheaper than solution - and (frontier[0])[0] < solution
        if debug:
            print("\n new node: ")
            print("frontier: ", frontier)
        (cost, node) = frontier.pop()
        if debug:
            print("cost: ", cost)
            print("node: ", node)
            print("node [-1]: ", node[-1])
        if node[-1] == goal:
            if show_result:
                print("frontier: ", frontier)
                print("reached goal: ", node)
                print("final cost: ", cost)
            return node
        explored_nodes.append(node[-1])
        
        for neighbor in graph.neighbors(node[-1]):
            frontier_paths = [el[1] for el in frontier]
            frontier_list = [item for sublist in frontier_paths for item in sublist]
            #print("frontier paths: ", frontier_paths)
            if debug:
                print("new neighbor: \n")
                print("neighbor: ", neighbor)
                print("frontier: ", frontier)
            if neighbor not in explored_nodes and neighbor not in frontier_list:
                amount = graph.get_edge_weight(node[-1], neighbor)
                new_cost = cost + amount
                new_path = list(node)
                new_path.append(neighbor)
                frontier.append((new_cost, new_path))
                if debug:
                    print("amount: ", amount)
                    print("new_cost: ", new_cost)
                    print("new path: ", new_path)
                    print("new frontier: ", frontier)

            elif neighbor in frontier_list:
                amount = graph.get_edge_weight(node[-1], neighbor)
                new_cost = cost + amount
                new_path = list(node)
                new_path.append(neighbor)
                if debug:
                    print("new cost: ", new_cost)
                    print("new path: ", new_path)
                    print("\n")
                    print("length of frontier before count starts: ", frontier.size())
                
                for item in frontier:
                    item_cost, path = item
                    if debug:
                        print("")
                        print("neighbor: ", neighbor)
                        print("path - 1", path[-1])
                        print("item: ", item)
                        print("item cost: ", item_cost)
                        print("item path: ", path)
                        print("new cost: ", new_cost)

                    if neighbor == path[-1]:
                        previous_cost = item_cost
                        if debug:
                            print("previous cost: ", previous_cost)
                        
                        if new_cost < previous_cost:
                            if debug:
                                print("\n****** removal and add to frontier ****")
                                print("remove from this frontier: ", frontier)
                                print("path looking to remove: ", path)
                            frontier = frontier.remove_if_same_path(path)
                            frontier.append((new_cost, new_path))
                            if debug:
                                print("add to frontier: ", (new_cost, new_path))
                                print("frontier after element add: ", frontier)

    return []
    #raise NotImplementedError


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    #print("v: ", v)
    #print("goal: ", goal)
    #print("graph.node[n]['pos'], ", graph.node[v]['pos'])

    if v == goal:
        return 0

    #print("graph node v: ", graph.node[v])
    posv = graph.node[v]['pos']
    posgoal = graph.node[goal]['pos']
    #print("value position ", posv)
    #print("goal position ", posgoal)

    distance = math.sqrt(math.pow(posv[0]-posgoal[0],2) + math.pow(posv[1]-posgoal[1],2))
    #print("distance: ", distance)
    return distance
    #raise NotImplementedError



def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    #test = heuristic(graph, start, goal)
    #print("test: ", test)


   
    #raise NotImplementedError

    

    debug = False
    show_result = False

    if start == goal:
        return []
    
    frontier = PriorityQueue()
    frontier.append((0, [start]))
    explored_nodes = []    

    while frontier: # and top(frontier) is cheaper than solution - and (frontier[0])[0] < solution
        if debug:
            print("\n new node: ")
            print("frontier: ", frontier)
        (cost, node) = frontier.pop()
        if debug:
            print("cost: ", cost)
            print("node: ", node)
            print("node [-1]: ", node[-1])
        if node[-1] == goal:
            if show_result:
                print("frontier: ", frontier)
                print("reached goal: ", node)
                print("final cost: ", cost)
            return node
        explored_nodes.append(node[-1])
        for neighbor in graph.neighbors(node[-1]):
            frontier_paths = [el[1] for el in frontier]
            #print("frontier paths: ", frontier_paths)
            if debug:
                print("new neighbor: \n")
                print("neighbor: ", neighbor)
                print("frontier: ", frontier)
            frontier_list = [item for sublist in frontier_paths for item in sublist]
            if neighbor not in explored_nodes and neighbor not in frontier_list:
                amount = graph.get_edge_weight(node[-1], neighbor)
                h_cost = heuristic(graph, neighbor, goal)
                old_cost = heuristic(graph, node[-1], goal)
                new_cost = cost + amount + h_cost - old_cost
                new_path = list(node)
                new_path.append(neighbor)
                frontier.append((new_cost, new_path))
                
                if debug:
                    print("amount: ", amount)
                    print("new_cost: ", new_cost)
                    print("new path: ", new_path)
                    print("new frontier: ", frontier)

            elif neighbor in frontier_list:
                amount = graph.get_edge_weight(node[-1], neighbor)
                h_cost = heuristic(graph, neighbor, goal)
                old_cost = heuristic(graph, node[-1], goal)
                new_cost = cost + amount + h_cost - old_cost
                new_path = list(node)
                new_path.append(neighbor)
                if debug:
                    print("new cost: ", new_cost)
                    print("new path: ", new_path)
                    print("\n")                          

                for item in frontier:
                    item_cost, path = item

                    if neighbor == path[-1]:
                        previous_cost = item_cost
                        if debug:
                            print("previous cost: ", previous_cost)
                        if new_cost < previous_cost:
                            frontier = frontier.remove_if_same_path(path)
                            frontier.append((new_cost, new_path))

    return []




def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """



    
    debug = False  
    show_result = False


    if start == goal:
        return []

    frontier_forward = PriorityQueue()
    frontier_forward.append((0, [start]))
    explored_nodes_forward = []
    explored_paths_forward = {}

    frontier_backward = PriorityQueue()
    frontier_backward.append((0, [goal]))
    explored_nodes_backward = []
    explored_paths_backward = {}

    paths_found = []
    full_path_cost = float('inf')

    fwd_frontier = True
    min_forward_cost = 0
    min_backward_cost = 0

    fwd_frontier_turn = True

    #print("hotlanta")
    #print("start: ", start)
    #print("goal: ", goal)


    while frontier_forward.size() > 0 and frontier_backward.size() > 0:

        if debug:
            print("______________________________")
            print("Starting new iteration")
            print("frontier forward: ", frontier_forward)
            print("frontier_backward: ", frontier_backward)
            print("explored_nodes_forward: ", explored_nodes_forward)
            print("explored_nodes_backward: ", explored_nodes_backward)
            print("explored_paths_forward: ", explored_paths_forward)
            print("explored_paths_backward: ", explored_paths_backward)
            print("paths found: ", paths_found)
            print("full_path_cost: ", full_path_cost)
            print("min forward cost: ", min_forward_cost)
            print("min backward cost: ", min_backward_cost)


        # stopping condition
        if min_forward_cost + min_backward_cost > full_path_cost:
            if show_result:
                print("******** shortest path found: *****", full_path_cost, " : ", full_path_path)
                print("forward direction: ", fwd_frontier_turn)
                print("start: ", start)
                print("goal: ", goal)
                print("complete_path_found: ", full_path_path)
                print("min forward cost: ", min_forward_cost)
                print("min backward cost: ", min_backward_cost)
                print("full path cost: ", full_path_cost)
            return full_path_path

        # decide if take step forward or backward
        # if min_forward_cost <= min_backward_cost:
        #     fwd_frontier = True
        # else:
        #     fwd_frontier = False

        if fwd_frontier_turn:
            (cost_forward, node_forward) = frontier_forward.pop()
            this_node_forward = node_forward[-1]
            
            #check if at goal
            if this_node_forward == goal:
                if show_result:
                    print("frontier: ", frontier_forward)
                    print("goal: ", goal)
                    print("final path: ", node_forward)
                    print("final cost: ", cost_forward)
                if len(paths_found)>0:
                    return full_path_path
                else:
                    return node_forward
            
            # append node to explored nodes forward
            explored_nodes_forward.append(this_node_forward)

            # append to explored paths forward
            explored_paths_forward[this_node_forward] = (cost_forward, node_forward)
            
            # check if node is in explored backward          
            if this_node_forward in explored_paths_backward.keys():
                (bkw_cost, bkw_path) = explored_paths_backward[this_node_forward]
                if debug: 
                    print(" ")
                    print("---------- found an explored path ------ ")
                    print("forward cost: ", cost_forward)
                    print("forward path: ", node_forward)
                    print("explored_paths_backward cost: ", bkw_cost)
                    print("explored_paths_backward node: ", bkw_path)
                    print("explored_paths_backward: ", explored_paths_backward)
                    print("fronter backward: ", frontier_backward)
                whole_path = []
                fake_path = []
                for item in node_forward:
                    fake_path.append(item)
                #whole_path.append(new_path_forward)
                reverse = bkw_path[::-1]
                for item in reverse:
                    fake_path.append(item)
                for x in fake_path:
                    if x not in whole_path:
                        whole_path.append(x)
                if debug:
                    print("whole path", whole_path)
                #whole_path = [y for x in whole_path for y in x]
                whole_cost = cost_forward+bkw_cost
                paths_found.append((whole_cost, whole_path))
                paths_found.sort(key=lambda x: x[0])
                full_path_cost = (paths_found[0])[0]
                full_path_path = (paths_found[0])[1]
                #return full_path_path


                # check pohls if any other shorter paths
                for pohl_node in explored_paths_forward.keys():
                    if pohl_node != this_node_forward:
                        if pohl_node in explored_paths_forward.keys():
                            (fwd_cost, fwd_path) = explored_paths_forward[pohl_node]
                        for item in frontier_backward:
                            if pohl_node == (item[1])[-1]:
                                bkw_cost = item[0]
                                bkw_path = item[1]
                                whole_path = []
                                fake_path = []
                                for item in fwd_path:
                                    fake_path.append(item)
                                reverse = bkw_path[::-1]
                                for item in reverse:
                                    fake_path.append(item)
                                for x in fake_path:
                                    if x not in whole_path:
                                        whole_path.append(x)
                                #whole_path = [y for x in whole_path for y in x]
                                whole_cost = bkw_cost + fwd_cost
                                paths_found.append((whole_cost, whole_path))
                                paths_found.sort(key=lambda x: x[0])
                                full_path_cost = (paths_found[0])[0]
                                full_path_path = (paths_found[0])[1]
                                if debug:
                                    print("\n POHLS Method ----")
                                    print("fwd_path: ", fwd_path)
                                    print("bkw_path: ", bkw_path[::-1])
                                    print("bkw_path 1: ", bkw_path[1::-1])
                                    print("pohl_node: ", pohl_node)
                                    print("forward cost: ", fwd_cost)
                                    print("backward cost: ", bkw_cost)
                                    print("whole path: ", whole_path)
                                    print("whole cost: ", whole_cost)


            

            # get latest forward frontier nodes
            frontier_paths_forward = [el[1] for el in frontier_forward]
            frontier_list_forward = [item for sublist in frontier_paths_forward for item in sublist]
            #if debug:
                #print("frontier_paths_forward: ", frontier_paths_forward)
                #print("frontier_list_forward: ", frontier_list_forward)

            #check if node in frontier
            #if this_node_forward in frontier_list_forward:


            # start looping over neighbors
            if debug:
                print("start looping over neighbors")

            for neighbor in graph.neighbors(this_node_forward):
                no_match = True
                
                #check if node is start and neighbor is goal
                # if this_node_forward == start and neighbor == goal:
                #     new_path_forward = []
                #     new_path_forward.append(this_node_forward)
                #     new_path_forward.append(neighbor)
                #     if show_result:
                #         print("straight line goal between 2 points: ", start, goal, " the end")
                #         print("final path: ", new_path_forward)
                #     return new_path_forward


                # check if neighbor already in explored or frontier
                if neighbor not in explored_nodes_forward and neighbor not in frontier_list_forward:
                
                    # neighbor not in either, add to frontier
                    amount = graph.get_edge_weight(this_node_forward, neighbor)
                    new_cost_forward  = cost_forward + amount
                    new_path_forward = list(node_forward)
                    new_path_forward.append(neighbor)
                    frontier_forward.append((new_cost_forward, new_path_forward))
                    if debug:
                        print("neighbor not in explored or frontier. Added to frontier: ", (new_cost_forward, new_path_forward))
                    
                #
                elif neighbor in frontier_list_forward:
                    amount = graph.get_edge_weight(this_node_forward, neighbor)
                    new_cost_forward  = cost_forward + amount
                    new_path_forward = list(node_forward)
                    new_path_forward.append(neighbor)
                    if debug:
                        print("neighbor in frontier. check if this is better path")

                    for item_forward in frontier_forward:
                        (item_cost_forward, path_forward) = item_forward

                        if neighbor == path_forward[-1]:
                            previous_cost_forward  = item_cost_forward 
                            if new_cost_forward  < previous_cost_forward:
                                frontier_forward = frontier_forward.remove_if_same_path(path_forward)
                                frontier_forward.append((new_cost_forward, new_path_forward))
                                if debug:
                                    print("new path was better: ", (new_cost_forward, new_path_forward))
                                    print("replaced: ", (previous_cost_forward, path_forward))
                            else:
                                if debug:
                                    print("previous path was better, not replaced", (new_cost_forward, new_path_forward))

                
                # check if neighbor is in backward path explored. If so add to complete path 
                """
                if neighbor in explored_paths_backward.keys():
                    (bkw_cost, bkw_path) = explored_paths_backward[neighbor]
                    whole_path = []
                    fake_path = []
                    for item in node_forward:
                        fake_path.append(item)
                    #whole_path.append(new_path_forward)
                    reverse = bkw_path[::-1]
                    for item in reverse:
                        fake_path.append(item)
                    for x in fake_path:
                        if x not in whole_path:
                            whole_path.append(x)
                    if debug:
                        print("whole path", whole_path)
                    whole_path = [y for x in whole_path for y in x]
                    whole_cost = cost_forward+bkw_cost
                    paths_found.append((whole_cost, whole_path))
                    paths_found.sort(key=lambda x: x[0])
                    full_path_cost = (paths_found[0])[0]
                    full_path_path = (paths_found[0])[1]
                   """
                                   
                    
                    
                # find new minimum of frontier forward
                
                if frontier_forward.size() > 0:
                    min_forward_cost = frontier_forward.top()[0]
                      
            fwd_frontier_turn = False


        else:
            if debug:
                print("___________________________________________________")
                print("\nbegin backward iteration\n")
            
            (cost_backward, node_backward) = frontier_backward.pop()
            this_node_backward = node_backward[-1]
            #if debug:
                #print("cost backward: ", cost_backward)
                #print("path backward: ", node_backward)
                #print("this_node_backward: ", this_node_backward)
            #check if at start
            if this_node_backward == start:
                if show_result:
                    print("frontier: ", frontier_backward)
                    print("start: ", start)
                    print("final path: ", node_backward)
                    print("final cost: ", cost_backward)
                    print("yes this one")
                    #print("ful path: ", full_path_path)
                    print("paths_found: ", paths_found)
                if len(paths_found)>0:
                    return full_path_path
                else:
                    whole_path = []
                    reverse = node_backward[::-1]
                    for item in reverse:
                        whole_path.append(item)
                    return whole_path

            
            # append node to explored nodes backward
            explored_nodes_backward.append(this_node_backward)

            # append to explored paths forward
            explored_paths_backward[this_node_backward] = (cost_backward, node_backward)
          
            # check if node is in explored backward        
            if this_node_backward in explored_paths_forward.keys():
                (fwd_cost, fwd_path) = explored_paths_forward[this_node_backward]
                if debug:
                    print(" ")
                    print("---------- Found an explored path in Backward Iteration ------ ")
                    print("this node: ", node_backward)
                    print("")
                    print("explored_paths_forward cost: ", fwd_cost)
                    print("explored_paths_forward node: ", fwd_path)
                    print("\nexplored paths backward: ", explored_paths_backward)
                    print("\nfrontier backward: ", frontier_backward)
                    print("\nexplored_paths_forward: ", explored_paths_forward)
                    print("\nfrontier fowrad: ", frontier_forward)
                whole_path = []
                fake_path = []
                for item in fwd_path:
                    fake_path.append(item)
                #whole_path.append(new_path_forward)
                reverse = node_backward[::-1]
                for item in reverse:
                    fake_path.append(item)
                for x in fake_path:
                    if x not in whole_path:
                        whole_path.append(x)
                #whole_path = [y for x in whole_path for y in x]
                whole_cost = cost_backward+fwd_cost
                paths_found.append((whole_cost, whole_path))
                paths_found.sort(key=lambda x: x[0])
                full_path_cost = (paths_found[0])[0]
                full_path_path = (paths_found[0])[1]
                if debug:
                    print("complete path: ", whole_path)
                    print("complete path cost: ", whole_cost)
                    print("")
                #return full_path_path


                # check pohls if any other shorter paths
                for pohl_node in explored_paths_backward.keys():
                    if pohl_node != this_node_backward:
                        if pohl_node in explored_paths_backward.keys():
                            (bkw_cost, bkw_path) = explored_paths_backward[pohl_node]
                        for item in frontier_forward:
                            if pohl_node == (item[1])[-1]:
                                fwd_cost = item[0]
                                fwd_path = item[1]
                                whole_path = []
                                fake_path = []
                                for item in fwd_path:
                                    fake_path.append(item)
                                #print("")
                                #print("fake_path: ", fake_path)
                                reverse = bkw_path[::-1]
                                #print("")
                                #print("reverse: ", reverse)
                                for item in reverse:
                                    fake_path.append(item)
                                for x in fake_path:
                                    if x not in whole_path:
                                        whole_path.append(x)
                                #print("")
                                #print("whole_path before y for x: ", whole_path)
                                #whole_path = [y for x in whole_path for y in x]
                                #print("")
                                #print("whole_path afterrrrr y for x: ", whole_path)
                                whole_cost = bkw_cost + fwd_cost
                                paths_found.append((whole_cost, whole_path))
                                paths_found.sort(key=lambda x: x[0])
                                full_path_cost = (paths_found[0])[0]
                                full_path_path = (paths_found[0])[1]
                                if show_result:
                                    print("\n POHLS Method ----")
                                    print("fwd_path: ", fwd_path)
                                    print("bkw_path: ", bkw_path[::-1])
                                    print("bkw_path 1: ", bkw_path[1::-1])
                                    print("pohl_node: ", pohl_node)
                                    print("forward cost: ", fwd_cost)
                                    print("backward cost: ", bkw_cost)
                                    print("whole path: ", whole_path)
                                    print("whole cost: ", whole_cost)







            #if debug:
                #print("explored nodes: ", explored_nodes_backward)

            frontier_paths_backward = [el[1] for el in frontier_backward]
            frontier_list_backward = [item for sublist in frontier_paths_backward for item in sublist]
            #if debug:
                #print("frontier_paths_backward: ", frontier_paths_backward)
                #print("frontier_list_backward: ", frontier_list_backward)
            
            # start looping over neighbors
            if debug:
                print("start looping over neighbors")
            
            for neighbor in graph.neighbors(this_node_backward):
                no_match = True
                if debug:
                    print ("\n next neighbor: ", neighbor)
                
                #check if only 2 neighbor of start node is goal
                # if this_node_backward == goal and neighbor == start:
                #     new_path_backward = []
                #     new_path_backward.append(neighbor)
                #     new_path_backward.append(goal)
                #     if show_result:
                #         print("straight line goal between 2 points: ", start, goal, " the end")
                #         print("final path: ", new_path_backward)
                #     return new_path_backward
                
                if neighbor not in explored_nodes_backward and neighbor not in frontier_list_backward:
                #if neighbor not in explored_nodes and neighbor not in frontier_list_backward:
                    amount = graph.get_edge_weight(this_node_backward, neighbor)
                    new_cost_backward  = cost_backward + amount
                    new_path_backward = list(node_backward)
                    new_path_backward.append(neighbor)
                    frontier_backward.append((new_cost_backward, new_path_backward))
                    if debug:
                        print("neighbor not in explored or frontier. Added to frontier: ", (new_cost_backward, new_path_backward))
                    


                elif neighbor in frontier_list_backward:
                    amount = graph.get_edge_weight(this_node_backward, neighbor)
                    new_cost_backward  = cost_backward + amount
                    new_path_backward = list(node_backward)
                    new_path_backward.append(neighbor)
                    if debug:
                        print("neighbor in frontier. check if this is better path")

                    for item_backward in frontier_backward:
                        (item_cost_backward, path_backward) = item_backward

                        if neighbor == path_backward[-1]:
                            previous_cost_backward  = item_cost_backward 
                            if new_cost_backward  < previous_cost_backward:
                                frontier_backward = frontier_backward.remove_if_same_path(path_backward)
                                frontier_backward.append((new_cost_backward, new_path_backward))
                                if debug:
                                    print("new path was better: ", (new_cost_backward, new_path_backward))
                                    print("replaced: ", (previous_cost_backward, path_backward))
                            else:
                                if debug:
                                    print("previous path was better, not replaced", (new_cost_backward, new_path_backward))
                    
                

                """
                if neighbor in explored_paths_forward.keys():
                    (fwd_cost, fwd_path) = explored_paths_forward[neighbor]
                    if debug:
                        print(" ")
                        print("---------- we found an explored path already ------ ")
                        print("explored_paths_forward cost: ", fwd_cost)
                        print("explored_paths_forward node: ", fwd_path)
                    whole_path = []
                    fake_path = []
                    for item in fwd_path:
                        fake_path.append(item)
                    #whole_path.append(new_path_forward)
                    reverse = node_backward[::-1]
                    for item in reverse:
                        fake_path.append(item)
                    for x in fake_path:
                        if x not in whole_path:
                            whole_path.append(x)
                    whole_path = [y for x in whole_path for y in x]
                    whole_cost = cost_backward+fwd_cost
                    paths_found.append((whole_cost, whole_path))
                    paths_found.sort(key=lambda x: x[0])
                    full_path_cost = (paths_found[0])[0]
                    full_path_path = (paths_found[0])[1]
                """
                
                        

            if frontier_backward.size()>0:
                #print("\nfrontier_backward: ", frontier_backward)
                #print("frontier size: ", frontier_backward.size())
                min_backward_cost = frontier_backward.top()[0]

            fwd_frontier_turn = True

        
    if len(paths_found)>0:
        if show_result:
            print("forward frontier: ", frontier_forward)
            print("start: ", start)
            print("goal: ", goal)
            print(" finished while, return: ", (paths_found[0])[1])
            print("complete paths: ", paths_found)
            #print("back frontier: ", frontier_backward)
            print(" ------ end -------\n")
        return (paths_found[0])[1]
    else:
        if show_result:
            print("forward frontier: ", frontier_forward)
            print("start: ", start)
            print("goal: ", goal)
            print("complete paths: ", paths_found)
            answer = []
            print("result: ", answer)
            #print("back frontier: ", frontier_backward)
            print(" ------ end -------\n")
        return[]



def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """


    #print statements = 1, no print = 0
    debug = False  
    show_result = True


    if start == goal:
        return []

    frontier_forward = PriorityQueue()
    frontier_forward.append((0, [start]))
    explored_nodes_forward = []
    explored_paths_forward = {}

    frontier_backward = PriorityQueue()
    frontier_backward.append((0, [goal]))
    explored_nodes_backward = []
    explored_paths_backward = {}

    paths_found = []
    full_path_cost = float('inf')

    fwd_frontier = True
    min_forward_cost = 0
    min_backward_cost = 0

    fwd_frontier_turn = True

    full_h_cost = heuristic(graph, start, goal)

    if show_result:
        print("")
        print("----new test---")
        print("hotlanta")
        print("start: ", start)
        print("goal: ", goal)


    while frontier_forward.size() > 0 and frontier_backward.size() > 0:

        if debug:
            print("______________________________")
            print("Starting new iteration")
            print("frontier forward: ", frontier_forward)
            print("frontier_backward: ", frontier_backward)
            print("explored_nodes_forward: ", explored_nodes_forward)
            print("explored_nodes_backward: ", explored_nodes_backward)
            print("explored_paths_forward: ", explored_paths_forward)
            print("explored_paths_backward: ", explored_paths_backward)
            print("paths found: ", paths_found)
            print("full_path_cost: ", full_path_cost)
            print("min forward cost: ", min_forward_cost)
            print("min backward cost: ", min_backward_cost)


        # stopping condition
        if min_forward_cost + min_backward_cost - full_h_cost > full_path_cost:
            if show_result:
                print("******** shortest path found: *****", full_path_cost, " : ", full_path_path)
                print("forward direction: ", fwd_frontier_turn)
                print("start: ", start)
                print("goal: ", goal)
                print("complete_path_found: ", full_path_path)
                print("min forward cost: ", min_forward_cost)
                print("min backward cost: ", min_backward_cost)
                print("full path cost: ", full_path_cost)
            return full_path_path

        if fwd_frontier_turn:
            (cost_forward, node_forward) = frontier_forward.pop()
            this_node_forward = node_forward[-1]
            
            #check if at goal
            if this_node_forward == goal:
                if show_result:
                    print("frontier: ", frontier_forward)
                    print("goal: ", goal)
                    print("final path: ", node_forward)
                    print("final cost: ", cost_forward)
                if len(paths_found)>0:
                    return full_path_path
                else:
                    return node_forward
            
            # append node to explored nodes forward
            explored_nodes_forward.append(this_node_forward)

            # append to explored paths forward
            explored_paths_forward[this_node_forward] = (cost_forward, node_forward)
            
            # check if node is in explored backward          
            if this_node_forward in explored_paths_backward.keys():
                (bkw_cost, bkw_path) = explored_paths_backward[this_node_forward]
                if debug: 
                    print(" ")
                    print("---------- found an explored path ------ ")
                    print("forward cost: ", cost_forward)
                    print("forward path: ", node_forward)
                    print("explored_paths_backward cost: ", bkw_cost)
                    print("explored_paths_backward node: ", bkw_path)
                    print("explored_paths_backward: ", explored_paths_backward)
                    print("fronter backward: ", frontier_backward)
                whole_path = []
                fake_path = []
                for item in node_forward:
                    fake_path.append(item)
                #whole_path.append(new_path_forward)
                reverse = bkw_path[::-1]
                for item in reverse:
                    fake_path.append(item)
                for x in fake_path:
                    if x not in whole_path:
                        whole_path.append(x)
                if debug:
                    print("whole path", whole_path)
                #whole_path = [y for x in whole_path for y in x]

                cost_forward_extra = heuristic(graph, this_node_forward, goal)
                cost_backward_extra = heuristic(graph, bkw_path[-1], start)

                if bkw_path[-1] == goal:
                    cost_backward_extra = 0
                if this_node_forward == start:
                    cost_forward_extra=0

                whole_cost = cost_forward+bkw_cost-cost_forward_extra-cost_backward_extra
                #whole_cost = cost_forward+bkw_cost
                paths_found.append((whole_cost, whole_path))
                paths_found.sort(key=lambda x: x[0])
                full_path_cost = (paths_found[0])[0]
                full_path_path = (paths_found[0])[1]
                #return full_path_path


                # check pohls if any other shorter paths
                for pohl_node in explored_paths_forward.keys():
                    if pohl_node != this_node_forward:
                        if pohl_node in explored_paths_forward.keys():
                            (fwd_cost, fwd_path) = explored_paths_forward[pohl_node]
                        for item in frontier_backward:
                            if pohl_node == (item[1])[-1]:
                                bkw_cost = item[0]
                                bkw_path = item[1]
                                whole_path = []
                                fake_path = []
                                for item in fwd_path:
                                    fake_path.append(item)
                                reverse = bkw_path[::-1]
                                for item in reverse:
                                    fake_path.append(item)
                                for x in fake_path:
                                    if x not in whole_path:
                                        whole_path.append(x)
                                #whole_path = [y for x in whole_path for y in x]

                                cost_forward_extra = heuristic(graph, fwd_path[-1], goal)
                                cost_backward_extra = heuristic(graph, bkw_path[-1], start)

                                if bkw_path[-1] == goal:
                                    cost_backward_extra = 0
                                if fwd_path[-1] == start:
                                    cost_forward_extra=0

                                whole_cost = fwd_cost+bkw_cost-cost_forward_extra-cost_backward_extra

                                #whole_cost = bkw_cost + fwd_cost
                                paths_found.append((whole_cost, whole_path))
                                paths_found.sort(key=lambda x: x[0])
                                full_path_cost = (paths_found[0])[0]
                                full_path_path = (paths_found[0])[1]
                                if debug:
                                    print("\n POHLS Method ----")
                                    print("fwd_path: ", fwd_path)
                                    print("bkw_path: ", bkw_path[::-1])
                                    print("bkw_path 1: ", bkw_path[1::-1])
                                    print("pohl_node: ", pohl_node)
                                    print("forward cost: ", fwd_cost)
                                    print("backward cost: ", bkw_cost)
                                    print("whole path: ", whole_path)
                                    print("whole cost: ", whole_cost)
                                    print("cost forward extra: ", cost_forward_extra)
                                    print("cost backward extra: ", cost_backward_extra)


            

            # get latest forward frontier nodes
            frontier_paths_forward = [el[1] for el in frontier_forward]
            frontier_list_forward = [item for sublist in frontier_paths_forward for item in sublist]


            # start looping over neighbors

            for neighbor in graph.neighbors(this_node_forward):
                no_match = True

                # check if neighbor already in explored or frontier
                if neighbor not in explored_nodes_forward and neighbor not in frontier_list_forward:
                
                    # neighbor not in either, add to frontier
                    amount = graph.get_edge_weight(this_node_forward, neighbor)

                    h_cost = heuristic(graph, neighbor, goal)
                    if this_node_forward == start:
                        old_cost = 0
                    else:
                        old_cost = heuristic(graph, this_node_forward, goal)
                    new_cost_forward = cost_forward + amount + h_cost - old_cost

                    if debug:
                        print("node: ", this_node_forward)
                        print("neighbor: ", neighbor)
                        print("cost forward: ", cost_forward)
                        print("amount: ", amount)
                        print("h_cost: ", h_cost)
                        print("old_cost: ", old_cost)
                        print("new cost forward: ", new_cost_forward)
                        print("")

                    #new_cost_forward  = cost_forward + amount
                    new_path_forward = list(node_forward)
                    new_path_forward.append(neighbor)
                    frontier_forward.append((new_cost_forward, new_path_forward))
                    if debug:
                        print("neighbor not in explored or frontier. Added to frontier: ", (new_cost_forward, new_path_forward))
                    
                #
                elif neighbor in frontier_list_forward:
                    amount = graph.get_edge_weight(this_node_forward, neighbor)

                    h_cost = heuristic(graph, neighbor, goal)
                    if this_node_forward == start:
                        old_cost = 0
                    else:
                        old_cost = heuristic(graph, this_node_forward, goal)
                    new_cost_forward = cost_forward + amount + h_cost - old_cost

                    #new_cost_forward  = cost_forward + amount
                    new_path_forward = list(node_forward)
                    new_path_forward.append(neighbor)

                    for item_forward in frontier_forward:
                        (item_cost_forward, path_forward) = item_forward

                        if neighbor == path_forward[-1]:
                            previous_cost_forward  = item_cost_forward 
                            if new_cost_forward  < previous_cost_forward:
                                frontier_forward = frontier_forward.remove_if_same_path(path_forward)
                                frontier_forward.append((new_cost_forward, new_path_forward))
                                if debug:
                                    print("new path was better: ", (new_cost_forward, new_path_forward))
                                    print("replaced: ", (previous_cost_forward, path_forward))
                            else:
                                if debug:
                                    print("previous path was better, not replaced", (new_cost_forward, new_path_forward))
                
                if frontier_forward.size() > 0:
                    min_forward_cost = frontier_forward.top()[0]
                      
            fwd_frontier_turn = False


        else:
            if debug:
                print("___________________________________________________")
                print("\nbegin backward iteration\n")
            
            (cost_backward, node_backward) = frontier_backward.pop()
            this_node_backward = node_backward[-1]

            #check if at start
            if this_node_backward == start:
                if show_result:
                    print("frontier: ", frontier_backward)
                    print("start: ", start)
                    print("final path: ", node_backward)
                    print("final cost: ", cost_backward)
                    print("yes this one")
                    #print("ful path: ", full_path_path)
                    print("paths_found: ", paths_found)
                if len(paths_found)>0:
                    return full_path_path
                else:
                    whole_path = []
                    reverse = node_backward[::-1]
                    for item in reverse:
                        whole_path.append(item)
                    return whole_path

            
            # append node to explored nodes backward
            explored_nodes_backward.append(this_node_backward)

            # append to explored paths forward
            explored_paths_backward[this_node_backward] = (cost_backward, node_backward)
          
            # check if node is in explored backward        
            if this_node_backward in explored_paths_forward.keys():
                (fwd_cost, fwd_path) = explored_paths_forward[this_node_backward]
                if debug:
                    print(" ")
                    print("---------- Found an explored path in Backward Iteration ------ ")
                    print("this node: ", node_backward)
                    print("")
                    print("explored_paths_forward cost: ", fwd_cost)
                    print("explored_paths_forward node: ", fwd_path)
                    print("\nexplored paths backward: ", explored_paths_backward)
                    print("\nfrontier backward: ", frontier_backward)
                    print("\nexplored_paths_forward: ", explored_paths_forward)
                    print("\nfrontier fowrad: ", frontier_forward)
                whole_path = []
                fake_path = []
                for item in fwd_path:
                    fake_path.append(item)
                reverse = node_backward[::-1]
                for item in reverse:
                    fake_path.append(item)
                for x in fake_path:
                    if x not in whole_path:
                        whole_path.append(x)

                cost_forward_extra = heuristic(graph, fwd_path[-1], goal)
                cost_backward_extra = heuristic(graph, this_node_backward, start)

                if this_node_backward == goal:
                    cost_backward_extra = 0
                if fwd_path[-1] == start:
                    cost_forward_extra=0

                whole_cost = cost_backward+fwd_cost-cost_forward_extra-cost_backward_extra

                #whole_cost = cost_backward+fwd_cost
                paths_found.append((whole_cost, whole_path))
                paths_found.sort(key=lambda x: x[0])
                full_path_cost = (paths_found[0])[0]
                full_path_path = (paths_found[0])[1]
                if debug:
                    print("complete path: ", whole_path)
                    print("complete path cost: ", whole_cost)
                    print("")

                # check pohls if any other shorter paths
                for pohl_node in explored_paths_backward.keys():
                    if pohl_node != this_node_backward:
                        if pohl_node in explored_paths_backward.keys():
                            (bkw_cost, bkw_path) = explored_paths_backward[pohl_node]
                        for item in frontier_forward:
                            if pohl_node == (item[1])[-1]:
                                fwd_cost = item[0]
                                fwd_path = item[1]
                                whole_path = []
                                fake_path = []
                                for item in fwd_path:
                                    fake_path.append(item)
                                reverse = bkw_path[::-1]
                                for item in reverse:
                                    fake_path.append(item)
                                for x in fake_path:
                                    if x not in whole_path:
                                        whole_path.append(x)


                                cost_forward_extra = heuristic(graph, fwd_path[-1], goal)
                                cost_backward_extra = heuristic(graph, bkw_path[-1], start)
                                if bkw_path[-1] == goal:
                                    cost_backward_extra = 0
                                if fwd_path[-1] == start:
                                    cost_forward_extra=0

                                whole_cost = fwd_cost+bkw_cost-cost_forward_extra-cost_backward_extra

                                #whole_cost = bkw_cost + fwd_cost
                                paths_found.append((whole_cost, whole_path))
                                paths_found.sort(key=lambda x: x[0])
                                full_path_cost = (paths_found[0])[0]
                                full_path_path = (paths_found[0])[1]
                                if show_result:
                                    print("\n POHLS Method ----")
                                    print("fwd_path: ", fwd_path)
                                    print("bkw_path: ", bkw_path[::-1])
                                    print("bkw_path 1: ", bkw_path[1::-1])
                                    print("pohl_node: ", pohl_node)
                                    print("forward cost: ", fwd_cost)
                                    print("backward cost: ", bkw_cost)
                                    print("whole path: ", whole_path)
                                    print("whole cost: ", whole_cost)
                                    print("cost forward extra: ", cost_forward_extra)
                                    print("cost backward extra: ", cost_backward_extra)

            frontier_paths_backward = [el[1] for el in frontier_backward]
            frontier_list_backward = [item for sublist in frontier_paths_backward for item in sublist]
            
            # start looping over neighbors
            if debug:
                print("start looping over neighbors")
            
            for neighbor in graph.neighbors(this_node_backward):
                no_match = True
                if debug:
                    print ("\n next neighbor: ", neighbor)
                
                if neighbor not in explored_nodes_backward and neighbor not in frontier_list_backward:
                    amount = graph.get_edge_weight(this_node_backward, neighbor)

                    h_cost = heuristic(graph, neighbor, start)
                    if this_node_backward == goal:
                        old_cost = 0
                    else:
                        old_cost = heuristic(graph, this_node_backward, start)
                    new_cost_backward = cost_backward + amount + h_cost - old_cost

                    #new_cost_backward  = cost_backward + amount
                    new_path_backward = list(node_backward)
                    new_path_backward.append(neighbor)
                    frontier_backward.append((new_cost_backward, new_path_backward))
                    
                elif neighbor in frontier_list_backward:
                    amount = graph.get_edge_weight(this_node_backward, neighbor)

                    h_cost = heuristic(graph, neighbor, start)
                    if this_node_backward == goal:
                        old_cost = 0
                    else:
                        old_cost = heuristic(graph, this_node_backward, start)
                    new_cost_backward = cost_backward + amount + h_cost - old_cost

                    #new_cost_backward  = cost_backward + amount
                    new_path_backward = list(node_backward)
                    new_path_backward.append(neighbor)

                    for item_backward in frontier_backward:
                        (item_cost_backward, path_backward) = item_backward

                        if neighbor == path_backward[-1]:
                            previous_cost_backward  = item_cost_backward 
                            if new_cost_backward  < previous_cost_backward:
                                frontier_backward = frontier_backward.remove_if_same_path(path_backward)
                                frontier_backward.append((new_cost_backward, new_path_backward))
            if frontier_backward.size()>0:
                #print("\nfrontier_backward: ", frontier_backward)
                #print("frontier size: ", frontier_backward.size())
                min_backward_cost = frontier_backward.top()[0]

            fwd_frontier_turn = True

        
    if len(paths_found)>0:
        if show_result:
            print("forward frontier: ", frontier_forward)
            print("start: ", start)
            print("goal: ", goal)
            print(" finished while, return: ", (paths_found[0])[1])
            print("complete paths: ", paths_found)
            #print("back frontier: ", frontier_backward)
            print(" ------ end -------\n")
        return (paths_found[0])[1]
    else:
        if show_result:
            print("forward frontier: ", frontier_forward)
            print("start: ", start)
            print("goal: ", goal)
            print("complete paths: ", paths_found)
            answer = []
            print("result: ", answer)
            #print("back frontier: ", frontier_backward)
            print(" ------ end -------\n")
        return[]
    


