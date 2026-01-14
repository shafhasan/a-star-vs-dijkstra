import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
import math
from matplotlib import animation

# ----------------------------
# Step 1: Generate Maze
# ----------------------------
def generate_maze(rows, cols, wall_prob=0.3, seed=42):
    np.random.seed(seed)
    return np.random.choice([0, 1], size=(rows, cols), p=[1 - wall_prob, wall_prob])


# ----------------------------
# Step 2: Heuristic Functions
# ----------------------------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def octile(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

def weighted_manhattan(a, b, weight=1.2):
    return weight * (abs(a[0] - b[0]) + abs(a[1] - b[1]))

def zero(a, b):
    return 0  # Dijkstra


# ----------------------------
# Step 3: Heuristic Selection
# ----------------------------
def select_heuristic():
    print("\nChoose a heuristic function:")
    print("1. Manhattan (4-direction)")
    print("2. Euclidean (8-direction)")
    print("3. Chebyshev (8-direction, equal cost)")
    print("4. Octile (8-direction, diagonal √2 cost)")
    print("5. Zero (Dijkstra-style)")

    choice = input("Enter your choice (1-5): ").strip()
    if choice == "1":
        return manhattan, "Manhattan"
    elif choice == "2":
        return euclidean, "Euclidean"
    elif choice == "3":
        return chebyshev, "Chebyshev"
    elif choice == "4":
        return octile, "Octile"
    elif choice == "5":
        return zero, "Zero (Dijkstra)"
    else:
        print("Invalid choice. Using Manhattan by default.")
        return manhattan, "Manhattan"


# ----------------------------
# Step 4: A* Algorithm
# ----------------------------
def astar_with_traversal(maze, start, goal, heuristic_func, heuristic_name):
    rows, cols = maze.shape
    open_list = []
    heapq.heappush(open_list, (heuristic_func(start, goal), 0, start, [start]))  # (f, g, node, path)
    visited = set()
    traversal_order = []
    cost = 0
    start_time = time.time()

    # Allow diagonal moves if heuristic is 8-directional
    if heuristic_name in ["Manhattan"]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-direction
    else:
        directions = [  # 8-direction
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

    while open_list:
        f, g, (r, c), path = heapq.heappop(open_list)
        if (r, c) in visited:
            continue
        visited.add((r, c))
        traversal_order.append((r, c))
        cost += 1

        if (r, c) == goal:
            total_time = time.time() - start_time
            return path, traversal_order, cost, total_time

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0 and (nr, nc) not in visited:
                step_cost = math.sqrt(2) if abs(dr) + abs(dc) == 2 else 1
                new_g = g + step_cost
                new_f = new_g + heuristic_func((nr, nc), goal)
                heapq.heappush(open_list, (new_f, new_g, (nr, nc), path + [(nr, nc)]))

    total_time = time.time() - start_time
    return None, traversal_order, cost, total_time


# ----------------------------
# Step 5: Animation
# ----------------------------
def animate_maze(maze, traversal, path, start, goal, filename="astar_animation.mp4"):
    traversal = list(traversal)
    path = list(path) if path else []

    rows, cols = maze.shape
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(maze, cmap='gray_r', origin='upper', vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    start_scatter = ax.scatter(start[1], start[0], s=120, color='green', edgecolor='black', zorder=5)
    goal_scatter = ax.scatter(goal[1], goal[0], s=120, color='blue', edgecolor='black', zorder=5)
    explored_scatter = ax.scatter([], [], s=40, color='yellow', alpha=0.4, zorder=2)
    path_scatter = ax.scatter([], [], s=60, color='red', zorder=6)

    def offsets(points):
        if not points: return np.empty((0, 2))
        return np.array([(c, r) for (r, c) in points])

    total_frames = len(traversal) + len(path)

    def update(frame):
        if frame < len(traversal):
            explored_scatter.set_offsets(offsets(traversal[:frame+1]))
        else:
            path_progress = frame - len(traversal)
            path_scatter.set_offsets(offsets(path[:path_progress]))
        return explored_scatter, path_scatter

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=70, repeat=False)

    try:
        ani.save(filename, writer='ffmpeg', fps=12)
        print("🎥 Saved animation:", filename)
    except Exception:
        gif = filename.replace(".mp4", ".gif")
        ani.save(gif, writer='pillow', fps=12)
        print("🎞 Saved animation:", gif)

    plt.show()


# ----------------------------
# Step 6: Main Program
# ----------------------------
# def main():
#     rows = int(input("Enter number of rows: "))
#     cols = int(input("Enter number of columns: "))
#     wall_prob = float(input("Enter wall probability (0-1, e.g. 0.3): "))
#
#     maze = generate_maze(rows, cols, wall_prob)
#     print("Generated Maze (0 = open, 1 = wall):\n", maze)
#
#     def parse_position(text):
#         return tuple(map(int, text.replace("(", "").replace(")", "").replace(",", " ").split()))
#
#     while True:
#         start = parse_position(input("Enter start position (row,col): "))
#         goal = parse_position(input("Enter goal position (row,col): "))
#         if maze[start] == 0 and maze[goal] == 0:
#             break
#         print("❌ Start or goal is on a wall. Choose open cells (0s).")
#
#     heuristic, h_name = select_heuristic()
#     path, traversal, cost, total_time = astar_with_traversal(maze, start, goal, heuristic, h_name)
#
#     print(f"\n--- A* Algorithm ({h_name} Heuristic) ---")
#     print(f"✅ Path found: {bool(path)}")
#     print(f"🧭 Nodes explored: {cost}")
#     print(f"⏱ Time: {total_time:.6f} s")
#
#     if path:
#         animate_maze(maze, traversal, path, start, goal, filename=f"astar_{h_name}.mp4")
#     else:
#         print("❌ No path found.")
#         plt.imshow(maze, cmap='gray_r', origin='upper')
#         plt.scatter(start[1], start[0], color='green', s=100)
#         plt.scatter(goal[1], goal[0], color='blue', s=100)
#         plt.show()
#


def main():
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))
    wall_prob = float(input("Enter wall probability (0-1, e.g. 0.3): "))

    maze = generate_maze(rows, cols, wall_prob)
    print("Generated Maze (0 = open, 1 = wall):\n", maze)

    def parse_position(text):
        return tuple(map(int, text.replace("(", "").replace(")", "").replace(",", " ").split()))

    while True:
        start = parse_position(input("Enter start position (row,col): "))
        goal = parse_position(input("Enter goal position (row,col): "))
        if maze[start] == 0 and maze[goal] == 0:
            break
        print("❌ Start or goal is on a wall. Choose open cells (0s).")

    while True:
        heuristic, h_name = select_heuristic()
        path, traversal, cost, total_time = astar_with_traversal(maze, start, goal, heuristic, h_name)

        print(f"\n--- A* Algorithm ({h_name} Heuristic) ---")
        print(f"✅ Path found: {bool(path)}")
        print(f"🧭 Nodes explored: {cost}")
        print(f"⏱ Time: {total_time:.6f} s")

        if path:
            animate_maze(maze, traversal, path, start, goal, filename=f"astar_{h_name}.mp4")
        else:
            print("❌ No path found.")
            plt.imshow(maze, cmap='gray_r', origin='upper')
            plt.scatter(start[1], start[0], color='green', s=100)
            plt.scatter(goal[1], goal[0], color='blue', s=100)
            plt.show()

        # Ask user if they want to run again
        choice = input("\nDo you want to run again? (y/n): ").strip().lower()
        if choice != 'y':
            print("👋 Exiting program. Goodbye!")
            break

# ----------------------------
if __name__ == "__main__":
    main()
