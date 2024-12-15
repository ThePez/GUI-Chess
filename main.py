# Gui chess game
from typing import Type
import uuid

ROWS = 8
COLUMNS = 8


class Engine:
    def __init__(self) -> None:
        self.board: list[list[BasePiece | None]] | None = None
        self.last_move: tuple[dict[str: tuple[int, int]], BasePiece] | None = None
        self.king_positions: dict[str: tuple[int, int]] | None = None
        self.players: list[str] = ["white", "black"]
        # Fill the board with the pieces
        self.setup_game()

    def setup_game(self) -> None:
        colours: list[str] = ["white", "black"]
        self.board = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]  # initialise the board structure
        self.king_positions = {"black": (0, 4), "white": (7, 4)}
        self.last_move = None
        for colour in colours:
            self.place_pieces(colour, is_pawn=False)  # Place other pieces
            self.place_pieces(colour, is_pawn=True)  # Place pawns

    def place_pieces(self, colour: str, is_pawn: bool) -> None:
        # White: specials go on row 7, pawns go on row 6
        # Black specials go on row 0, pawns go on row 1
        assert colour in ["white", "black"]
        if is_pawn:
            row: int = 6 if colour == "white" else 1
            pieces: list[Type[BasePiece]] = [Pawn] * COLUMNS
        else:
            row: int = 7 if colour == "white" else 0
            pieces: list[Type[BasePiece]] = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]

        for col, piece_type in enumerate(pieces):
            piece: BasePiece = piece_type((row, col), colour)
            self.board[row][col] = piece

    def print_board(self) -> None:
        width: int = 16  # makes the words appear centered, min is 15
        # Header for column labels
        print(f"{' ' * 3} {'a':^{width}} {'b':^{width}} {'c':^{width}} {'d':^{width}} {'e':^{width}} ", end="")
        print(f"{'f':^{width}} {'g':^{width}} {'h':^{width}}")
        print(f"{' ' * 2}{'+'}{''.join(['-' * width + '+' for _ in range(8)])}")
        for row_index, row in enumerate(self.board):
            print(f"{8 - row_index} |", end="")  # Row label (chess numbering)
            for piece in row:
                piece_str = str(piece) if piece else " " * width  # Empty squares are blank
                print(f"{piece_str:^{width}}|", end="")

            print(f" {8 - row_index}")  # Row label (chess numbering)
            print(f"{' ' * 2}{'+'}{''.join(['-' * width + '+' for _ in range(8)])}")  # Divider after each row

        # Footer for column labels
        print(f"{' ' * 3} {'a':^{width}} {'b':^{width}} {'c':^{width}} {'d':^{width}} {'e':^{width}} ", end="")
        print(f"{'f':^{width}} {'g':^{width}} {'h':^{width}}")

    def get_piece(self, row: int, col: int) -> 'BasePiece':
        return self.board[row][col]

    def turn_checker(self) -> str:
        if not self.last_move:
            # White goes first
            return "white"

        if self.last_move[1].colour == "black":
            return "white"
        else:
            return "black"

    def promote_pawn(self, position: tuple[int, int]) -> bool:
        row, col = position
        pawn: BasePiece = self.get_piece(row, col)
        colour = pawn.colour
        if isinstance(pawn, Pawn):
            if (colour == "white" and row == 0) or (colour == "black" and row == 7):
                print("Pawn promotion! Choose a piece:")
                print("1. Queen")
                print("2. Rook")
                print("3. Bishop")
                print("4. Knight")
                try:
                    choice: int = int(input("Enter the number of your choice: "))

                except (ValueError, TypeError):
                    print("Invalid input! Defaulting to Queen.")
                    choice = 1

                promoted_piece: BasePiece | None = None
                if choice == 1:
                    promoted_piece = Queen(position, colour)
                elif choice == 2:
                    promoted_piece = Rook(position, colour)
                elif choice == 3:
                    promoted_piece = Bishop(position, colour)
                elif choice == 4:
                    promoted_piece = Knight(position, colour)
                else:
                    print("Invalid choice. Defaulting to Queen.")
                    promoted_piece = Queen(position, colour)

                # Replace the pawn with the chosen piece
                self.board[row][col] = promoted_piece
                return True

        return False

    def promote_checker(self) -> None:
        for row in (self.board[0], self.board[7]):
            for piece in row:
                if piece:
                    self.promote_pawn(piece.position)

    def castling_checker(self, colour: str, target_position: tuple[int, int]) -> bool:
        king_position: tuple[int, int] = self.king_positions[colour]
        king: BasePiece | None = self.get_piece(king_position[0], king_position[1])
        row: int = king_position[0]
        target_col: int = target_position[1]
        king_col: int = king_position[1]
        if target_col > king_col:  # King side castling
            rook_position: tuple[int, int] = (row, 7)
            rook_col: int  = 7
            rook_target_col: int = 5
        else:  # Queen side castling
            rook_position: tuple[int, int] = (row, 0)
            rook_col: int = 0
            rook_target_col: int = 3

        rook: BasePiece | None = self.get_piece(row, rook_position[1])
        if rook is None or not rook.never_moved or king is None or not king.never_moved:
            return False

        # Check squares between are empty
        for col in range(min(rook_col, king_col) + 1, max(rook_col, king_col)):
            if self.get_piece(row, col) is not None:
                return False

        # Check that the king is not in check is any of the squares it moves through
        for col in [king_col, target_col, rook_target_col]:
            if self.check_test(king_position, (row, col)):
                return False

        # Checks have passed, now move both pieces
        self._move_piece(rook_position, (row, rook_target_col))
        self._move_piece(king_position, target_position)
        return True

    def en_passant_test(self, start_position: tuple[int, int], target_position: tuple[int, int]) -> bool:
        piece: BasePiece | None = self.get_piece(start_position[0], start_position[1])
        # Should be a pawn at (start_row, end_col)
        target_piece: BasePiece | None = self.get_piece(start_position[0], target_position[1])
        if piece is None or target_piece is None:
            return False

        if not isinstance(piece, Pawn) or not isinstance(target_piece, Pawn):
            # Can only do the En-passant move with 2 pawns
            return False

        if self.last_move:  # Can only do the en-passant move on the next turn
            last_move_start: tuple[int, int] = self.last_move[0]["start"]
            last_move_end: tuple[int, int] = self.last_move[0]["end"]
            last_move_piece: 'BasePiece | None' = self.last_move[1]
            if not isinstance(last_move_piece, Pawn):
                # Last move must have been a pawn being moved
                return False

            if (abs(last_move_end[0] - last_move_start[0]) == 2
                    and last_move_end == (start_position[0], target_position[1])):
                # Pawns were are adjacent
                return True

        return False

    def en_passant_king_test(self, pieces: tuple['BasePiece', 'BasePiece'], end: tuple[int, int]) -> bool:
        attacking_pawn = pieces[0]
        captured_pawn = pieces[1]
        capture_row, capture_col = captured_pawn.position
        self.board[capture_row][capture_col] = None
        check_test: bool = self.check_test(attacking_pawn.position, end)
        self.board[capture_row][capture_col] = captured_pawn
        return check_test

    def en_passant_move(self, start_position: tuple[int, int], target_position: tuple[int, int]) -> None:
        self._move_piece(start_position, target_position)
        self.board[start_position[0]][target_position[1]] = None

    def _move_piece(self, start_position: tuple[int, int], target_position: tuple[int, int]) -> None:
        # assumes that there is a piece at start_position and that the move is valid.
        start_row, start_col = start_position
        end_row, end_col = target_position
        piece: BasePiece = self.get_piece(start_row, start_col)
        self.board[start_row][start_col] = None
        self.board[end_row][end_col] = piece
        piece.update_position(target_position)
        self.last_move = ({"start": start_position, "end": target_position}, piece)
        if isinstance(piece, King):
            self.king_positions[piece.colour] = target_position

    def check_test(self, start_position: tuple[int, int], target_position: tuple[int, int]) -> bool:
        start_row, start_col = start_position
        target_row, target_col = target_position
        piece: BasePiece = self.get_piece(start_row, start_col)
        if isinstance(piece, King):
            self.king_positions[piece.colour] = target_position
        target_piece: BasePiece | None = self.get_piece(target_row, target_col)
        self.board[start_row][start_col] = None
        self.board[target_row][target_col] = piece
        check_test: bool = self.is_king_in_check(piece.colour)
        self.board[start_row][start_col] = piece
        self.board[target_row][target_col] = target_piece
        if isinstance(piece, King):
            self.king_positions[piece.colour] = start_position
        return check_test

    def is_king_in_check(self, king_colour: str):
        king_position: tuple[int, int] = self.king_positions[king_colour]
        for row in self.board:
            for piece in row:
                if piece and piece.colour != king_colour:
                    if piece.check_move(self.board, king_position) and piece.capture_test(self.board, king_position):
                        return True

        return False

    def validate_move(self, start_position: tuple[int, int], target_position: tuple[int, int]) -> bool:
        start_row, start_col = start_position
        target_row, target_col = target_position
        piece: BasePiece | None = self.board[start_row][start_col]
        target_piece: BasePiece | None = self.board[target_row][target_col]
        if piece is None:
            return False  # Need to be moving a piece.

        if piece.colour != self.turn_checker():
            print(f"Not {piece.colour}'s turn.")
            return False

        if isinstance(target_piece, King):
            # Can't take a king
            return False

        if isinstance(piece, King):
            delta_col = abs(target_col - start_col)
            if delta_col == 2 and self.castling_checker(piece.colour, target_position):
                return True

        if isinstance(piece, Pawn):
            # In here need to do the en-passant stuff as well as other pawn related stuff
            delta_col = abs(target_col - start_col)
            # Normal Pawn move
            if delta_col == 0 and piece.check_move(self.board, target_position):
                if not self.check_test(start_position, target_position):
                    # Pawn check move function will have already allowed a normal move
                    self._move_piece(start_position, target_position)
                    return True

            # Normal Pawn capture
            if piece.check_move(self.board, target_position) and not self.check_test(start_position, target_position):
                self._move_piece(start_position, target_position)
                return True

            # En-Passant Pawn capture
            if self.en_passant_test(start_position, target_position):
                pieces: tuple[BasePiece, BasePiece] = piece, self.get_piece(start_row, target_col)
                if not self.en_passant_king_test(pieces, target_position):
                    self.en_passant_move(start_position, target_position)
                    return True

            # Anything else is bad :(
            return False

        if not piece.check_move(self.board, target_position):
            return False  # The movement was bad -> return false

        if not piece.capture_test(self.board, target_position):
            return False

        if self.check_test(start_position, target_position):
            return False

        # All checks passed, thus move the piece and return true
        self._move_piece(start_position, target_position)
        return True

    def is_checkmate(self) -> bool:
        colours: list[str] = ["white", "black"]
        for colour in colours:
            threats: list[dict[str: tuple[int, int]]] = self.get_threats_to_king(colour)
            if len(threats) == 0:
                # No threats to this coloured king, so move on to other colour
                continue

            can_escape: bool = self.can_king_escape(colour)
            if len(threats) >= 2 and can_escape:
                continue

            can_block: bool = self.can_check_be_blocked(colour, threats)
            # can_block = False
            can_capture: bool = self.can_piece_be_captured(colour, threats)
            if can_block or can_capture or can_escape:
                continue

            # Unable to be blocked or piece captured -> King is checkmated
            return True

        # Either no threats found of preventable threats found -> not checkmate
        return False

    def get_threats_to_king(self, king_colour: str) -> list[dict[str: tuple[int, int]]]:
        threats: list[tuple[int, int]] = []
        king_position = self.king_positions[king_colour]
        opponent_colour: str = "white" if king_colour == "black" else "black"
        for row_pieces in self.board:
            for piece in row_pieces:
                if piece and piece.colour == opponent_colour and piece.check_move(self.board, king_position):
                    threats.append(piece.position)

        return threats

    def can_king_escape(self, king_colour: str) -> bool:
        moves: list[tuple[int, int]] = [(1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1)]
        king_position = self.king_positions[king_colour]
        king_row: int = king_position[0]
        king_col: int = king_position[1]
        king_piece: BasePiece = self.get_piece(king_row, king_col)
        for move in moves:
            new_row, new_col = king_row + move[0], king_col + move[1]
            new_position: tuple[int, int] = new_row, new_col
            if out_of_bounds((new_row, new_col)):
                continue

            if king_piece.check_move(self.board, new_position) and king_piece.capture_test(self.board, new_position):
                if not self.check_test(king_position, (new_row, new_col)):
                    return True

        return False

    def can_piece_be_captured(self, king_colour: str, threats: list[tuple[int, int]]) -> bool:
        for threat in threats:
            for row_pieces in self.board:
                for piece in row_pieces:
                    if piece and piece.colour == king_colour:
                        if (piece.check_move(self.board, threat)
                                and piece.capture_test(self.board, threat)
                                and not self.check_test(piece.position, threat)):
                            return True

        return False

    def can_check_be_blocked(self, king_colour: str, threats: list[tuple[int, int]]) -> bool:
        # threats has start: row, col and end: row, col
        # check if piece attacking is knight (as this can't be blocked)
        # work out change in rows and cols
        # store the list of these positions and find out if any piece can move into these spots
        # threats should be of length 1 here... I hope
        threat = threats[0]
        threat_row, threat_col = threat
        if isinstance(self.get_piece(threat_row, threat_col), Knight):
            # Can't block a knight's attack
            return False

        king_position = self.king_positions[king_colour]
        king_row, king_col = king_position
        delta_row = king_row - threat_row
        delta_col = king_col - threat_col
        step_row = 0 if delta_row == 0 else (1 if delta_row > 0 else -1)
        step_col = 0 if delta_col == 0 else (1 if delta_col > 0 else -1)
        block_squares: list[tuple[int, int]] = []
        current_row, current_col = threat_row + step_row, threat_col + step_col
        while (current_row, current_col) != king_position:
            block_squares.append((current_row, current_col))
            current_row += step_row
            current_col += step_col

        # now have the potential spaces to try and move pieces into
        return self.can_piece_be_captured(king_colour, block_squares)


class BasePiece:
    def __init__(self, position: tuple[int, int], colour: str) -> None:
        self.position = position
        self.colour = colour
        self.never_moved = True
        self.unique_id = uuid.uuid4()

    def __str__(self) -> str:
        return f"{self.colour.capitalize()} {self.__class__.__name__}"

    def update_position(self, new_position: tuple[int, int]) -> None:
        """
        Updates a piece's position with a new position. Will also update the piece's 'never_moved' attribute to false,
        as it has now been moved.
        :param new_position: tuple of new positional data (row, col)
        """
        self.position = new_position
        if self.never_moved:
            self.never_moved = False

    def check_move(self, board: list[list['BasePiece | None']], target_position: tuple[int, int]) -> bool:
        pass  # Each piece has specialised movement, thus this method is redefined in inherited classes

    def capture_test(self, board: list[list['BasePiece | None']], target_position: tuple[int, int]) -> bool:
        """
        Will determine if this piece is allowed to occupy the space pointed to by target_position. Will return true if
        the target location is either not of the same colour or empty ("None"). Kings are included here and will need
        to be excluded elsewhere.
        :param board: list of BasePiece's on the current game board
        :param target_position: (row, col) tuple
        :return: True or False
        """
        end_piece: 'BasePiece | None' = board[target_position[0]][target_position[1]]
        if end_piece and end_piece.colour == self.colour:
            return False  # Can't capture your own colours

        # End location is either emtpy or is a valid capture. Kings are excluded elsewhere.
        return True

    def path_checking(self, board: list[list['BasePiece | None']], target_position: tuple[int, int]) -> bool:
        """
        Function to check the path to a target position. It is assumed that the path is straight or diagonal.
        Will return False if any of the positions between the start and target are not "None", otherwise returns True.
        :param board: list of BasePiece's on the current game board
        :param target_position: (row, col) tuple
        :return: True or False
        """
        delta_row = target_position[0] - self.position[0]
        delta_col = target_position[1] - self.position[1]
        step_row = 0 if delta_row == 0 else (1 if delta_row > 0 else -1)
        step_col = 0 if delta_col == 0 else (1 if delta_col > 0 else -1)
        current_row, current_col = self.position[0] + step_row, self.position[1] + step_col
        # Will check the path taken by the move to ensure no pieces are blocking it. Doesn't check the final spot
        while (current_row, current_col) != target_position:
            if board[current_row][current_col] is not None:
                return False
            current_row, current_col = current_row + step_row, current_col + step_col

        return True


class Pawn(BasePiece):
    def check_move(self, board: list[list[BasePiece | None]], target_position: tuple[int, int], ) -> bool:
        delta_row = target_position[0] - self.position[0]
        delta_col = abs(target_position[1] - self.position[1])
        direction: int = 1 if self.colour == "black" else -1  # Pawns can only more forwards
        # Black moves down, white moves up
        if delta_col == 0:  # Straight movement
            if direction == delta_row:
                # Checking to make sure the target square is empty
                return board[target_position[0]][target_position[1]] is None

            if delta_row == 2 * direction and self.never_moved:
                # Check to make sure both squares are empty
                return (board[target_position[0] - direction][target_position[1]] is None
                        and board[target_position[0]][target_position[1]] is None)

            return False

        if delta_col == 1 and delta_row == direction:  # Capturing
            if self.capture_test(board, target_position):
                return True

            return False

        # Anything else is a bad move
        return False

    def capture_test(self, board: list[list[BasePiece | None]], target_position: tuple[int, int]) -> bool:
        target_piece: BasePiece | None = board[target_position[0]][target_position[1]]
        if target_piece and target_piece.colour != self.colour:
            return True

        return False


class Rook(BasePiece):
    def check_move(self, board: list[list['BasePiece | None']], target_position: tuple[int, int]) -> bool:
        delta_row = abs(target_position[0] - self.position[0])
        delta_col = abs(target_position[1] - self.position[1])
        # Can only move in straight lines
        if not min(delta_col, 1) ^ min(delta_row, 1):
            return False

        if not self.path_checking(board, target_position):
            return False

        return True


class Bishop(BasePiece):
    def check_move(self, board: list[list['BasePiece | None']], target_position: tuple[int, int]) -> bool:
        delta_row = abs(target_position[0] - self.position[0])
        delta_col = abs(target_position[1] - self.position[1])
        # Can only move in diagonals
        if delta_row != delta_col:
            return False

        if not self.path_checking(board, target_position):
            return False

        return True


class Knight(BasePiece):
    def check_move(self, board: list[list['BasePiece | None']], target_position: tuple[int, int]) -> bool:
        delta_row = abs(target_position[0] - self.position[0])
        delta_col = abs(target_position[1] - self.position[1])
        # L shaped movement, not path checking as knight can jump over other pieces
        if not ((delta_row == 2 and delta_col == 1) or (delta_row == 1 and delta_col == 2)):
            return False

        return True


class Queen(BasePiece):
    def check_move(self, board: list[list['BasePiece | None']], target_position: tuple[int, int]) -> bool:
        delta_row = abs(target_position[0] - self.position[0])
        delta_col = abs(target_position[1] - self.position[1])
        # Check the actual move... is it a valid movement?
        # Diagonals or straight lines only
        if not (delta_row == delta_col or min(delta_row, 1) ^ min(delta_col, 1)):
            return False

        # Check the path taken
        if not self.path_checking(board, target_position):
            return False

        # Checks passed -> valid move
        return True


class King(BasePiece):
    def check_move(self, board: list[list['BasePiece | None']], target_position: tuple[int, int]) -> bool:
        delta_row = abs(target_position[0] - self.position[0])
        delta_col = abs(target_position[1] - self.position[1])
        # Only one square at a time, so no path checking
        # Diagonals or straight lines only
        return False if not max(delta_row, delta_col) == 1 else True


def out_of_bounds(position: tuple[int, int]) -> bool:
    x, y = position
    return x < 0 or x > 7 or y < 0 or y > 7


def main_testing(engine: Engine) -> None:
    while True:
        engine.print_board()
        start = input("Start position:")
        end = input("End position:")
        if start.lower() == "quit" or end.lower() == "quit":
            break

        try:
            start_row = 8 - int(start[0])
            start_col = ord(start[1].lower()) - ord("a")
            start_pos = start_row, start_col
            end_row = 8 - int(end[0])
            end_col = ord(end[1].lower()) - ord("a")
            end_pos = end_row, end_col
        except (ValueError, IndexError, TypeError):
            print("Wrong input, try again")
            continue

        if not engine.validate_move(start_pos, end_pos):
            print("Bad move, try again...")
            continue

        if engine.is_checkmate():
            colour: str = "White" if engine.is_king_in_check('white') else "Black"
            engine.print_board()
            print(f"{colour} King has been checkmated, game over.")
            break

        engine.promote_checker()
        if engine.is_king_in_check("white"):
            print("White King is in Check")

        if engine.is_king_in_check("black"):
            print("Black King is in Check")

    user_input: str = input("Play again? (Yes/No) ")
    user_input.lower()
    if user_input == "yes":
        engine.setup_game()
        main_testing(engine)
    else:
        print("Goodbye!")
        return


if __name__ == '__main__':
    test_engine = Engine()
    main_testing(test_engine)
