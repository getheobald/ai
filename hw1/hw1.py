import time
import numpy as np
from gridgame import *

##############################################################################################################################

# You can visualize what your code is doing by setting the GUI argument in the following line to true.
# The render_delay_sec argument allows you to slow down the animation, to be able to see each step more clearly.

# For your final submission, please set the GUI option to False.

# The gs argument controls the grid size. You should experiment with various sizes to ensure your code generalizes.
# Please do not modify or remove lines 18 and 19.

##############################################################################################################################

game = ShapePlacementGrid(GUI=False, render_delay_sec=0.5, gs=6, num_colored_boxes=5)
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')
np.savetxt('initial_grid.txt', grid, fmt="%d")

##############################################################################################################################

# Initialization

# shapePos is the current position of the brush.

# currentShapeIndex is the index of the current brush type being placed (order specified in gridgame.py, and assignment instructions).

# currentColorIndex is the index of the current color being placed (order specified in gridgame.py, and assignment instructions).

# grid represents the current state of the board. 
    
    # -1 indicates an empty cell
    # 0 indicates a cell colored in the first color (indigo by default)
    # 1 indicates a cell colored in the second color (taupe by default)
    # 2 indicates a cell colored in the third color (veridian by default)
    # 3 indicates a cell colored in the fourth color (peach by default)

# placedShapes is a list of shapes that have currently been placed on the board.
    
    # Each shape is represented as a list containing three elements: a) the brush type (number between 0-8), 
    # b) the location of the shape (coordinates of top-left cell of the shape) and c) color of the shape (number between 0-3)

    # For instance [0, (0,0), 2] represents a shape spanning a single cell in the color 2=veridian, placed at the top left cell in the grid.

# done is a Boolean that represents whether coloring constraints are satisfied. Updated by the gridgames.py file.

##############################################################################################################################

shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')

print(shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done)


start = time.time()  # time code execution


def get_empty_cells(grid):
    # Returns array of empty cells, which are all cells with value of -1
    empty_cells = []
    gridSize = len(grid)
    for y in range(gridSize):
        for x in range(gridSize):
            if grid[y, x] == -1: # Numpy arrays are indexed row then column so has to be y then x
                empty_cells.append((x, y))
    return empty_cells

def is_illegal_cell(grid, x, y, color):
    # Check if a move is illegal (would cause there to be two same-colored cells adjacent to each other)
    # Get all neighbors of a given x,y position and check that they're not the same color as the given cell
    gridSize = len(grid)
    
    neighbors = [
        (x - 1, y),  # Left neighbor
        (x + 1, y),  # Right neighbor
        (x, y - 1),  # Top neighbor
        (x, y + 1)   # Bottom neighbor
    ]
    
    for neighbor in neighbors:
        x_value = neighbor[0]
        y_value = neighbor[1]
        # Make sure you're still in bounds
        if 0 <= x_value < gridSize and 0 <= y_value < gridSize:
            if grid[y_value, x_value] == color:
                return True # Same color = illegal move = true

    # No constraints broken, thus not an illegal move
    return False


def is_legal_move(grid, shape, pos, color):
    # Check if we can place a given shape in a given color at a given position without violating constraints
    # Returns true if shape physically fits on the board AND no cell in the shape would violate adjacency constraint

    # Does it fit
    if not game.canPlace(grid, shape, pos): 
        return False
    
    # Does it violate adjacency constraint
    # Used AI to generate this check
    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell == 1:
                x = pos[0] + j
                y = pos[1] + i
                
                # Check if this cell would violate constraint
                if is_illegal_cell(grid, x, y, color):
                    return False
    
    return True

def objective_function(grid, placedShapes):
    """
    Calculates the score of a move based on the optimization goals
    (I know this is a nontraditional function name but it helps me sort out what it's doing in my mind)
    
    A move gets a score by...
    * Maximizing cells filled, since we want to fill the grid
    * Minimizing shapes used
    * Minimizing colors used
    """    
    # Count filled cells
    filled_cells = np.sum(grid != -1)
    
    # Count shapes used
    num_shapes = len(placedShapes)
    
    # Count unique colors used
    # Used AI for this line - it uses numpy indexing and a set operation to first determine which cells are nonempty,
    # then get the color values of those cells, then use the set function to get the unique set
    unique_colors = set(grid[grid != -1]) 
    num_colors = len(unique_colors)
    
    # Calculate objective function score/value/output
    score = 0
    score += filled_cells * 10 # Primary goal is filling the grid, so this is most heavily rewarded
    score -= num_shapes
    score -= num_colors
    
    return score


def move_to(new_pos):
    # Helper func to move the brush to a given position

    global shapePos # Used AI to remember how global variables work and that I need to do this
    
    target_x, target_y = new_pos
    current_x, current_y = shapePos
    
    # Move right
    while current_x < target_x:
        game.execute('right')
        current_x += 1
    # Move left
    while current_x > target_x:
        game.execute('left')
        current_x -= 1
    # Move up
    while current_y > target_y:
        game.execute('up')
        current_y -= 1
    # Move down
    while current_y < target_y:
        game.execute('down')
        current_y += 1
    
    # Update position var
    shapePos = [target_x, target_y]


def switch_shape(new_shape):
    # Helper func to switch to the given shape
    # Shapes are numbered 0-8
    global currentShapeIndex
    while currentShapeIndex != new_shape:
        shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('switchshape')


def switch_color(new_color):
    # Helper func to switch to the given color
    # Colors are numbered 0-3
    global currentColorIndex
    while currentColorIndex != new_color:
        shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('switchcolor')


def place_shape(pos, shape, color):
    # Uses helpers to move to the given position, switch to the given shape and color,
    # and place a shape given those values

    global shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done
    
    move_to(pos)
    switch_shape(shape)
    switch_color(color)
    
    shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('place')


def first_choice_local_search():
    """
    Uses first-choice local search (hill climbing) to color an n x n grid, minimizing shapes and colors used
    Makes one attempt to fill the grid, maxing at 500 iterations or 50 failed moves, before admitting defeat
    """
    global shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done
    
    # Get objective func value of initial state
    current_score = objective_function(grid, placedShapes)

    # Set up safety checks    
    iterations = 0
    max_iterations = 500
    failed_moves = 0
    max_failed_moves = 50

    while not done and iterations < max_iterations:
        iterations += 1
                
        # Get empty cells
        empty_cells = get_empty_cells(grid)

        if not empty_cells:
            break
        
        # Generate random neighbor state
        random_pos = random.choice(empty_cells)
        random_shape = random.randint(0, 8)
        random_color = game.getAvailableColor(grid, random_pos[0], random_pos[1])
        
        # Get the actual shape
        shape = game.shapes[random_shape]
        
        # Check if this placement is legal, then do it
        if is_legal_move(grid, shape, random_pos, random_color):
            place_shape(random_pos, random_shape, random_color)
            
            # Check objective function for that move
            new_score = objective_function(grid, placedShapes)
            
            # Accept the move as long as it's the same or better
            if new_score >= current_score:
                current_score = new_score
                failed_moves = 0
                
            # If not same or better, undo that move
            else:
                shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('undo')
                failed_moves += 1
        else:
            failed_moves += 1
        
        if failed_moves >= max_failed_moves:
            return False

    # Used AI to generate these print statements to report on game state - unnecessary for submission
    filled = np.sum(grid != -1)
    total = len(grid) * len(grid)
    print(f"Grid filled: {filled} / {total} cells")
    print(f"Shapes used: {len(placedShapes)}")
    print(f"Colors used: {len(set(grid[grid != -1]))}")
    print(f"Constraints satisfied: {done}")
    
    if done:
        return True
    else:
        return False

def main():
    # Plays a game that fills an n x n grid with 4 different colors. Allows for 20 random restarts.

    global shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done

    for restart_attempt in range(20):
        # Call FCLS algorithm to try to fill the grid
        attempt = first_choice_local_search()
        if attempt:
            print("Success!")
            return
        # Attempt unsuccessful, so empty the grid by undoing all moves
        while len(placedShapes) > 0:
            shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('undo')


# Run
main()


end=time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))
