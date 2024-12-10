# Gui chess game
from typing import Type


class Engine:
    def __init__(self) -> None:
        self.board: list[list] = [[None for _ in range(8)] for _ in range(8)]  # initialise the board structure
        self.setup_board()  # Fill the board with the pieces

    def setup_board(self) -> None:
        colours: list[str] = ["white", "black"]
        for colour in colours:
            # self.place_pawns(colour)
            self.place_others(colour)

    def place_pawns(self, colour: str) -> None:
        if colour == 'white':
            for i in range(8):
                self.board[6][i] = Pawn((6, i), "white")

        if colour == 'black':
            for i in range(8):
                self.board[1][i] = Pawn((1, i), "black")

    def place_others(self, colour: str) -> None:
        pieces: list[Type[BasePiece]] = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        if colour == 'white':
            for i, Piece in enumerate(pieces):
                self.board[7][i] = Piece((7, i), "white")

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

    def move_piece(self, old_position: tuple[int, int], new_position: tuple[int, int]) -> None:
        old_row, old_col = old_position
        new_row, new_col = new_position
        if out_of_bounds(old_position) or out_of_bounds(new_position):
            print("Invalid board position!")
            return

        piece: 'BasePiece' = self.get_piece(old_row, old_col)
        if piece is None:
            print("Piece not found!")
            return

        if not piece.check_move(new_position, self.board):
            print("Invalid move!")
            return

        self.board[old_row][old_col] = None
        self.board[new_row][new_col] = piece
        piece.update_position(new_position)


class BasePiece:
    """
    White rows decrease
    Black rows increase
    """
    def __init__(self, position: tuple[int, int], colour: str) -> None:
        self.position = position
        self.colour = colour
        self.base = True

    def __str__(self) -> str:
        return f"{self.__class__.__name__} ({self.colour})"

    def update_position(self, position: tuple[int, int]) -> None:
        self.position = position
        self.base = False

    def check_move(self, new_position: tuple[int, int], board: list[list['BasePiece | None']]) -> bool:
        """
        Check if the requested move is valid for the type of piece. Does not care if pieces are in the way. Is only
        checking if the movement is valid.
        :param board: The game board, listing all active pieces
        :param new_position: new row and column position
        :return: bool
        """
        pass  # Each piece has specialised movement, thus this method is redefined in inherited classes


class Pawn(BasePiece):
    def check_move(self, new_position: tuple[int, int], board: list[list['BasePiece | None']]) -> bool:
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

            # logic for en-passant
            pass

        return False


class Rook(BasePiece):
    def check_move(self, new_position: tuple[int, int], board: list[list['BasePiece | None']]) -> bool:
        delta_row = new_position[0] - self.position[0]
        delta_col = new_position[1] - self.position[1]
        # Can only move in straight lines
        if not delta_col ^ delta_row or not path_checking(board, self.position, new_position):
            return False

        if not capture_test(board, new_position, self):
            return False

        return True


class Bishop(BasePiece):
    def check_move(self, new_position: tuple[int, int], board: list[list['BasePiece | None']]) -> bool:
        delta_row = abs(new_position[0] - self.position[0])
        delta_col = abs(new_position[1] - self.position[1])
        # Can only move in diagonals
        if delta_row != delta_col or not path_checking(board, self.position, new_position):
            return False

        if not capture_test(board, new_position, self):
            return False

        return True


class Knight(BasePiece):
    def check_move(self, new_position: tuple[int, int], board: list[list['BasePiece | None']]) -> bool:
        delta_row = abs(new_position[0] - self.position[0])
        delta_col = abs(new_position[1] - self.position[1])
        if not ((delta_row == 2 and delta_col == 1) or (delta_row == 1 and delta_col == 2)):
            return False

        if not capture_test(board, new_position, self):
            return False

        return True


class Queen(BasePiece):
    def check_move(self, new_position: tuple[int, int], board: list[list['BasePiece | None']]) -> bool:
        delta_row = abs(new_position[0] - self.position[0])
        delta_col = abs(new_position[1] - self.position[1])
        # Diagonals or straight lines only
        if not (delta_row == delta_col or delta_row ^ delta_col):
            return False

        if not path_checking(board, self.position, new_position):
            return False

        if not capture_test(board, new_position, self):
            return False

        return True


class King(BasePiece):
    def check_move(self, new_position: tuple[int, int], board: list[list['BasePiece | None']]) -> bool:
        delta_row = abs(new_position[0] - self.position[0])
        delta_col = abs(new_position[1] - self.position[1])
        # Only one square at a time
        # Diagonals or straight lines only
        if not max(delta_row, delta_col) == 1:
            return False

        if not capture_test(board, new_position, self):
            return False

        return True


def out_of_bounds(position: tuple[int, int]) -> bool:
    x, y = position
    return x < 0 or x > 7 or y < 0 or y > 7


def path_checking(board: list[list['BasePiece | None']], start: tuple[int, int], end: tuple[int, int]) -> bool:
    """
    Checks the path until the final spot. Final spot will need different logic.
    :param board:
    :param start:
    :param end:
    :return:
    """
    delta_row = end[0] - start[0]
    delta_col = end[1] - start[1]
    step_row = 0 if delta_row == 0 else (1 if delta_row > 0 else -1)
    step_col = 0 if delta_col == 0 else (1 if delta_col > 0 else -1)
    current_row, current_col = start[0] + step_row, start[1] + step_col
    while (current_row, current_col) != end:
        if board[current_row][current_col] is not None:
            return False
        current_row, current_col = current_row + step_row, current_col + step_col

    return True


def capture_test(board: list[list['BasePiece | None']], end: tuple[int, int], active_piece: 'BasePiece') -> bool:
    end_piece: 'BasePiece | None' = board[end[0]][end[1]]
    if end_piece is None:
        return True

    if end_piece.colour == active_piece.colour or isinstance(end_piece, King):
        return False  # Can't capture your own colours or capture a King

    return True  # Valid move


def main_testing(engine: Engine) -> None:
    engine.move_piece((0, 1), (2, 0))  # Knight
    engine.print_board()
    engine.move_piece((0, 3), (7, 3))  # Queen
    engine.print_board()


if __name__ == '__main__':
    test_engine = Engine()
    main_testing(test_engine)
