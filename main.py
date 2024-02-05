from wrapper import GameWrapperWario

if __name__ == "__main__":
    # Create a new instance of the wrapper
    game = GameWrapperWario("roms/wario_land.gb", window_type="SDL2", window_scale=3, debug=True, game_wrapper=True)

    # Start the game
    game.start_game()
