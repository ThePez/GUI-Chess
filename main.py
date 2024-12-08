# Gui chess game


class Engine:
    def __init__(self) -> None:
        self.board: list[list] = [[None for _ in range(8)] for _ in range(8)]
        self.setup_board()

    def setup_board(self) -> None:
        # Black side
        self.board[0][0] = Rook((0, 0))
        self.board[0][1] = Knight((0, 1))
        self.board[0][2] = Bishop((0, 2))
        self.board[0][3] = King((0, 3))
        self.board[0][4] = Queen((0, 4))
        self.board[0][5] = Bishop((0, 5))
        self.board[0][6] = Knight((0, 6))
        self.board[0][7] = Rook((0, 7))
        for i in range(8):
            self.board[1][i] = Pawn((1, i))

        # White Side
        self.board[7][0] = Rook((7, 0))
        self.board[7][1] = Knight((7, 1))
        self.board[7][2] = Bishop((7, 2))
        self.board[7][3] = Queen((7, 3))
        self.board[7][4] = King((7, 4))
        self.board[7][5] = Bishop((7, 5))
        self.board[7][6] = Knight((7, 6))
        self.board[7][7] = Rook((7, 7))
        for i in range(8):
            self.board[6][i] = Pawn((6, i))

    def print_board(self) -> None:
        for row in self.board:
            for piece in row:
                print(f"{str(piece):^6}", end=" ")
            print()  # new line
        print()

    def get_piece(self, row: int, col: int) -> 'BasePiece':
        return self.board[row][col]

    def move_piece(self, old_position: tuple[int, int], new_position: tuple[int, int]) -> None:
        row, col = old_position
        piece: 'BasePiece' = self.get_piece(row, col)
        piece.update_position(new_position)
        self.board[row][col] = None
        row, col = new_position
        self.board[row][col] = piece


class BasePiece:
    def __init__(self, position: tuple[int, int]) -> None:
        self.row = position[0]
        self.col = position[1]

    def update_position(self, position: tuple[int, int]) -> None:
        self.row = position[0]
        self.col = position[1]


class Pawn(BasePiece):
    def __init__(self, position: tuple[int, int]) -> None:
        super().__init__(position)

    def __str__(self) -> str:
        return f'Pawn'

    def check_move(self, delta_row: int) -> bool:
        pass


class Rook(BasePiece):
    def __init__(self, position: tuple[int, int]) -> None:
        super().__init__(position)

    def __str__(self) -> str:
        return f'Rook'


class Bishop(BasePiece):
    def __init__(self, position: tuple[int, int]) -> None:
        super().__init__(position)

    def __str__(self) -> str:
        return f'Bishop'


class Knight(BasePiece):
    def __init__(self, position: tuple[int, int]) -> None:
        super().__init__(position)

    def __str__(self) -> str:
        return f'Knight'


class Queen(BasePiece):
    def __init__(self, position: tuple[int, int]) -> None:
        super().__init__(position)

    def __str__(self) -> str:
        return f'Queen'


class King(BasePiece):
    def __init__(self, position: tuple[int, int]) -> None:
        super().__init__(position)

    def __str__(self) -> str:
        return f'King'


def main_testing(engine: Engine) -> None:
    engine.print_board()
    engine.move_piece((0, 1), (2, 0))
    engine.print_board()
    item = engine.get_piece(2, 0)
    print(f'{item} is at {item.row}:{item.col}')


if __name__ == '__main__':
    engine = Engine()
    main_testing(engine)
