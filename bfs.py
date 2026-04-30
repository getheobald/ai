from collections import deque

""""
def BFS (G, s):
	Q.enqueue( s )
	mark s as visited
	while (Q is not empty):
		v  =  Q.dequeue( )
		if v == goal:
			terminate
		for all neighbors w of v:
			if w not visited:
				if w == goal:
					terminate
				Q.enqueue( w )
				mark w as visited
    """

def bfs(graph, start, goal):
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        v, path = queue.popleft()
        if v == goal:
            print(path)
            return path
        for w in graph.get(v, []):
            if w not in visited:
                new_path = path + [w]
                if w == goal:
                    print(new_path)
                    return new_path
                queue.append((w, new_path))
                visited.add(w)
    print("No path found")
    return None

graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }

bfs(graph, 'A', 'F')

