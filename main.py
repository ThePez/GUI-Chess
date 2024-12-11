# Gui chess game
from typing import Type

ROWS = 8
COLUMNS = 8


class Engine:
    def __init__(self) -> None:
        self.board: list[list['BasePiece | None']] | None = None
        self.last_move: tuple[dict[str: dict[str: int]], 'BasePiece'] | None = None
        self.king_positions: dict[str: dict[str: int]] | None = None
        self.setup_board()  # Fill the board with the pieces

    def setup_board(self) -> None:
        self.board = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]  # initialise the board structure
        colours: list[str] = ["white", "black"]
        self.king_positions = {"black": {"row": 0, "column": 4}, "white": {"row": 7, "column": 4}}
        self.last_move = None
        for colour in colours:
            self.place_pawns(colour)
            self.place_others(colour)

    def place_pawns(self, colour: str) -> None:
        if colour == 'white':
            for i in range(COLUMNS):
                self.board[COLUMNS - 2][i] = Pawn((COLUMNS - 2, i), "white")

        if colour == 'black':
            for i in range(COLUMNS):
                self.board[1][i] = Pawn((1, i), "black")

    def place_others(self, colour: str) -> None:
        pieces: list[Type[BasePiece]] = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        if colour == 'white':
            for i, Piece in enumerate(pieces):
                self.board[COLUMNS - 1][i] = Piece((COLUMNS - 1, i), "white")

        if colour == 'black':
            for i, Piece in enumerate(pieces):
                self.board[0][i] = Piece((0, i), "black")

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

    def move_piece(self, old_position: tuple[int, int], new_position: tuple[int, int]) -> bool:
        old_row, old_col = old_position
        new_row, new_col = new_position
        if out_of_bounds(old_position) or out_of_bounds(new_position):
            print("Invalid board position!")
            return False

        piece: 'BasePiece' = self.get_piece(old_row, old_col)
        # Can't move nothing
        if piece is None:
            print("Piece not found!")
            return False

        # Check to ensure the requested move is valid
        if not piece.check_move(new_position, self.board, last_move=self.last_move):
            print("Invalid move!")
            return False

        # Check to ensure that king's can't be captured
        if isinstance(self.get_piece(new_row, new_col), King):
            print("Can't capture a King!")
            return False

        # Move piece to check for King check
        self.board[old_row][old_col] = None
        self.board[new_row][new_col] = piece
        opponent_colour = "white" if piece.colour == "black" else "black"
        if self.is_king_in_check(opponent_colour):
            # undo piece move
            self.board[old_row][old_col] = piece
            self.board[new_row][new_col] = None
            print("Can't put your King into check! or You need to protect your king")
            return False

        # If the piece being moved was a king, need to update the variable that stores the king locations
        if isinstance(piece, King):
            if piece.colour == "white":
                self.king_positions["white"] = {"row": new_row, "column": new_col}
            else:
                self.king_positions["black"] = {"row": new_row, "column": new_col}

        # Finally update the piece's internal position and store the last move
        piece.update_position(new_position)
        self.last_move = ({"start": {"row": old_row, "column": old_col},
                           "end": {"row": new_row, "column": new_col}},
                          piece)
        return True

    def is_king_in_check(self, opponent_colour: str):
        if opponent_colour == "white":
            king_position: tuple[int, int] = self.king_positions["black"]["row"], self.king_positions["black"]["column"]
        else:
            king_position: tuple[int, int] = self.king_positions["white"]["row"], self.king_positions["white"]["column"]

        for row in self.board:
            for piece in row:
                if piece and piece.colour == opponent_colour:
                    if piece.check_move(king_position, self.board, last_move=self.last_move):
                        return True

        return False

    def is_checkmate(self) -> bool:
        pass

    def get_threats_to_king(self, king_colour: str, opponent_colour: str) -> list[dict[str: int]]:
        threats: list[dict[str: int]] = []
        king_position = self.king_positions[king_colour]["row"], self.king_positions[king_colour]["column"]
        for row in self.board:
            for piece in row:
                if (piece and
                        piece.colour == opponent_colour and
                        piece.check_move(king_position, self.board, last_move=self.last_move)):
                    row, col = piece.position
                    threats.append({"row": row, "column": col})

        return threats

    def can_king_escape(self) -> bool:
        pass

    def can_piece_be_captured(self) -> bool:
        pass

    def can_check_be_blocked(self) -> bool:
        pass


class BasePiece:
    def __init__(self, position: tuple[int, int], colour: str) -> None:
        self.position = position
        self.colour = colour
        self.base = True

    def __str__(self) -> str:
        return f"{self.__class__.__name__} ({self.colour})"

    def update_position(self, position: tuple[int, int]) -> None:
        self.position = position
        if self.base:
            self.base = False

    def check_move(self,
                   new_position: tuple[int, int],
                   board: list[list['BasePiece | None']],
                   last_move: tuple[dict[str: dict[str: int]], 'BasePiece'] | None = None) -> bool:
        """
        Check if the requested move is valid for the type of piece. Does not care if pieces are in the way. Is only
        checking if the movement is valid.
        """
        pass  # Each piece has specialised movement, thus this method is redefined in inherited classes

    def capture_test(self, board: list[list['BasePiece | None']], end: tuple[int, int]) -> bool:
        end_piece: 'BasePiece | None' = board[end[0]][end[1]]
        if end_piece and end_piece.colour == self.colour:
            return False  # Can't capture your own colours

        # End location is either emtpy or is a valid capture. Kings are excluded elsewhere.
        return True

    def path_checking(self, board: list[list['BasePiece | None']], end: tuple[int, int]) -> bool:
        delta_row = end[0] - self.position[0]
        delta_col = end[1] - self.position[1]
        step_row = 0 if delta_row == 0 else (1 if delta_row > 0 else -1)
        step_col = 0 if delta_col == 0 else (1 if delta_col > 0 else -1)
        current_row, current_col = self.position[0] + step_row, self.position[1] + step_col
        # Will check the path taken by the move to ensure no pieces are blocking it. Doesn't check the final spot
        while (current_row, current_col) != end:
            if board[current_row][current_col] is not None:
                return False
            current_row, current_col = current_row + step_row, current_col + step_col

        return True


class Pawn(BasePiece):
    def check_move(self,
                   new_position: tuple[int, int],
                   board: list[list['BasePiece | None']],
                   last_move: tuple[dict[str: dict[str: int]], 'BasePiece'] | None = None) -> bool:
        delta_row = new_position[0] - self.position[0]
        delta_col = new_position[1] - self.position[1]
        direction: int = 1 if self.colour == "black" else -1
        # Black moves down, white moves up
        if delta_col == 0:  # Straight movement
            if direction == delta_row or (delta_row == 2 * direction and self.base):
                return board[new_position[0]][new_position[1]] is None

            return False

        if abs(delta_col) == 1 and delta_row == direction:  # Capturing
            target_piece: 'BasePiece | None' = board[new_position[0]][new_position[1]]
            # Normal Capture
            if target_piece and target_piece.colour != self.colour and not isinstance(target_piece, King):
                return True

            if last_move:  # logic for en-passant
                start: tuple[int, int] = last_move[0]["start"]["row"], last_move[0]["start"]["column"]
                end: tuple[int, int] = last_move[0]["end"]["row"], last_move[0]["end"]["column"]
                piece: 'BasePiece | None' = last_move[1]
                if not isinstance(piece, Pawn):
                    # Must be a Pawn
                    return False

                if abs(end[0] - start[0]) == 2 and end == (self.position[0], new_position[1]):
                    # Pawns are adjacent
                    board[end[0]][end[1]] = None  # Remove the captured pawn
                    return True

        # Anything else is a bad move
        return False


class Rook(BasePiece):
    def check_move(self,
                   new_position: tuple[int, int],
                   board: list[list['BasePiece | None']],
                   last_move: tuple[tuple[int, int], tuple[int, int], 'BasePiece'] | None = None) -> bool:
        delta_row = abs(new_position[0] - self.position[0])
        delta_col = abs(new_position[1] - self.position[1])
        # Can only move in straight lines
        if not min(delta_col, 1) ^ min(delta_row, 1):
            return False

        if not (self.path_checking(board, new_position) and self.capture_test(board, new_position)):
            return False

        return True


class Bishop(BasePiece):
    def check_move(self,
                   new_position: tuple[int, int],
                   board: list[list['BasePiece | None']],
                   last_move: tuple[tuple[int, int], tuple[int, int], 'BasePiece'] | None = None) -> bool:
        delta_row = abs(new_position[0] - self.position[0])
        delta_col = abs(new_position[1] - self.position[1])
        # Can only move in diagonals
        if delta_row != delta_col:
            return False

        if not (self.path_checking(board, new_position) and self.capture_test(board, new_position)):
            return False

        return True


class Knight(BasePiece):
    def check_move(self,
                   new_position: tuple[int, int],
                   board: list[list['BasePiece | None']],
                   last_move: tuple[tuple[int, int], tuple[int, int], 'BasePiece'] | None = None) -> bool:
        delta_row = abs(new_position[0] - self.position[0])
        delta_col = abs(new_position[1] - self.position[1])
        # L shaped movement, not path checking as knight jump other pieces
        if not ((delta_row == 2 and delta_col == 1) or (delta_row == 1 and delta_col == 2)):
            return False

        if not self.capture_test(board, new_position):
            return False

        return True


class Queen(BasePiece):
    def check_move(self,
                   new_position: tuple[int, int],
                   board: list[list['BasePiece | None']],
                   last_move: tuple[tuple[int, int], tuple[int, int], 'BasePiece'] | None = None) -> bool:
        delta_row = abs(new_position[0] - self.position[0])
        delta_col = abs(new_position[1] - self.position[1])
        # Check the actual move... is it a valid movement?
        # Diagonals or straight lines only
        if not (delta_row == delta_col or min(delta_row, 1) ^ min(delta_col, 1)):
            return False

        # Check the path taken, and if a capture is possible
        if not (self.path_checking(board, new_position) and self.capture_test(board, new_position)):
            return False

        # Checks passed -> valid move
        return True


class King(BasePiece):
    def check_move(self,
                   new_position: tuple[int, int],
                   board: list[list['BasePiece | None']],
                   last_move: tuple[tuple[int, int], tuple[int, int], 'BasePiece'] | None = None) -> bool:
        delta_row = abs(new_position[0] - self.position[0])
        delta_col = abs(new_position[1] - self.position[1])
        # Only one square at a time
        # Diagonals or straight lines only
        if not (max(delta_row, delta_col) == 1 and self.capture_test(board, new_position)):
            return False

        return True


def out_of_bounds(position: tuple[int, int]) -> bool:
    x, y = position
    return x < 0 or x > 7 or y < 0 or y > 7


def main_testing(engine: Engine) -> None:
    while True:
        engine.print_board()
        start = input("Start position as row col:").split()
        end = input("End position as row col:").split()
        # a = 97
        start_row = 8 - int(start[0])
        start_col = ord(start[1].lower()) - ord("a")
        start_pos = start_row, start_col
        end_row = 8 - int(end[0])
        end_col = ord(end[1].lower()) - ord("a")
        end_pos = end_row, end_col
        engine.move_piece(start_pos, end_pos)
        if engine.is_king_in_check('white'):
            print("Black King is in check")
        if engine.is_king_in_check('black'):
            print("White King is in check")


if __name__ == '__main__':
    test_engine = Engine()
    main_testing(test_engine)
