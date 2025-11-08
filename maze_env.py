


class MazeEnv:
    def __init__(self, layout):
        self.maze = layout
        self.start_pos = (1, 1)
        self.end_pos = (10, 10)

        self.height = len(self.maze)
        self.width = len(self.maze[0])

        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _find_char(self, char_code):
        for y, row in enumerate(self.maze):
            for x, tile in enumerate(row):
                if tile == char_code:
                    return (y, x)
        return None #won't happen

    def reset(self):
        return self.start_pos

    def step(self, current_pos_yx, action_int):

        if current_pos_yx == self.end_pos:
            return current_pos_yx, 0, True
        
        dy, dx = self.actions[action_int]
        new_y, new_x = current_pos_yx[0] + dy, current_pos_yx[1] + dx

        # bounds, edge of map 
        if not (0 <= new_y < self.height and 0 <= new_x < self.width):
            return current_pos_yx, -10, False

        # bounds, wall
        tile = self.maze[new_y][new_x]
        if tile == 1: # 1 = Wall
            return current_pos_yx, -10, False # Stay in place, penalize

        # Check for end
        if (new_y, new_x) == self.end_pos:
            return (new_y, new_x), 100, True # Reached end!!

        # banana
        if tile == 4:
            return current_pos_yx, 5, False
        
        # Valid floor move
        # (y, x), reward, done
        return (new_y, new_x), -1, False # -1 to encourage speed

    def get_layout(self):
        """Returns the static 2D maze layout."""
        return self.maze



if __name__ == "__main__":
    # 0 = FLOOR
    # 1 = WALL
    # 2 = START
    # 3 = END
    # 4 = BANANA
    
    
    TEST_MAZE = [
        [1, 1, 1, 1, 1],
        [1, 2, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 3, 1],
        [1, 1, 1, 1, 1]
    ]

    env = MazeEnv(TEST_MAZE)

    print(f"Maze Layout:\n{env.get_layout()}")

    pos = env.reset()
    print(f"Start pos: {pos}")

    # Try moving right
    pos, reward, done = env.step(pos, 3) # 3 = Right
    print(f"Move Right: Pos={pos}, Reward={reward}, Done={done}")
    
    # Try moving into wall
    pos, reward, done = env.step(pos, 2) # 2 = Left (back to start)
    pos, reward, done = env.step(pos, 0) # 0 = Up (into wall)
    print(f"Move Up (Wall): Pos={pos}, Reward={reward}, Done={done}")