# Gui chess game
from typing import Type
import sys
from contextlib import contextmanager
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QObject, pyqtSlot
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QMainWindow, QMessageBox, QFrame, QHBoxLayout)

ROWS = 8
COLUMNS = 8


class ViewController(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up the main window properties
        self.setWindowTitle("Chess by Jack :)")
        self.setGeometry(600, 50, 100, 100)

        # Start the engine
        self._engine = Engine()

        # Create a central widget to hold other widgets
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Turn Label
        self.turn_label = QLabel("White's Turn")
        self.turn_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.turn_label.setAlignment(Qt.AlignCenter)
        self.turn_label.setStyleSheet(
            "background-color: lightgray; padding: 5px; border: 1px solid black; border-radius: 5px; color: black;"
        )

        restart_button = QPushButton("Restart")
        restart_button.setFont(QFont("Arial", 12, QFont.Bold))
        restart_button.clicked.connect(self.restart_game)

        white_resign_button = QPushButton("White Resign")
        white_resign_button.setFont(QFont("Arial", 12, QFont.Bold))
        white_resign_button.clicked.connect(self.white_resigned)

        black_resign_button = QPushButton("Black Resign")
        black_resign_button.setFont(QFont("Arial", 12, QFont.Bold))
        black_resign_button.clicked.connect(self.black_resigned)
        resign_layout = QHBoxLayout()
        resign_layout.addWidget(white_resign_button)
        resign_layout.addWidget(black_resign_button)

        # Chessboard container
        self.board_container = QFrame()
        self.board_container.setFrameShape(QFrame.Box)
        self.board_container.setStyleSheet("border: 2px solid black; background-color: white;")
        self.board_layout = QGridLayout()
        self.board_layout.setSpacing(0)  # No gaps between buttons
        self.board_container.setLayout(self.board_layout)

        # Add row and column labels
        self.add_board_labels()

        # Set up the layout for the main window
        main_layout = QVBoxLayout()
        main_layout.addWidget(restart_button)
        main_layout.addWidget(self.turn_label)
        main_layout.addLayout(resign_layout)
        main_layout.addWidget(self.board_container)
        self.central_widget.setLayout(main_layout)

        # Variables used for running the game
        self.icons: dict[str: str] = {
            "White Pawn": "./icons/white-pawn.png",
            "Black Pawn": "./icons/black-pawn.png",
            "White Knight": "./icons/white-knight.png",
            "Black Knight": "./icons/black-knight.png",
            "White King": "./icons/white-king.png",
            "Black King": "./icons/black-king.png",
            "White Queen": "./icons/white-queen.png",
            "Black Queen": "./icons/black-queen.png",
            "White Rook": "./icons/white-rook.png",
            "Black Rook": "./icons/black-rook.png",
            "White Bishop": "./icons/white-bishop.png",
            "Black Bishop": "./icons/black-bishop.png",
        }
        self.first_button_press: tuple[int, int] | None = None
        self.board_buttons: list[list[QPushButton]] = []

        # Set up the UI
        self.make_board_buttons()
        self.update_buttons()
        self.setup_signals()

    def setup_signals(self) -> None:
        # Game ending Signals
        self._engine.checkmate.connect(self.game_over)
        self._engine.stalemate.connect(self.game_over)
        self._engine.insufficient.connect(self.game_over)
        # Player move signals
        self._engine.move_successful.connect(self.move_successful)
        self._engine.invalid_move.connect(self.move_failed)
        # Check signal
        self._engine.check_warning.connect(self.king_in_check)
        # Pawn Promotion signals
        self._engine.promote_pawn_signal.connect(self.get_promotion_choice)

        # self.promotion_choice.connect(self._engine.promote_pawn)
        # self.attempt_move.connect(self._engine.attempt_move)
        # self.check_check.connect(self._engine.check_for_check)
        # self.game_setup.connect(self._engine.setup_game)

    def add_board_labels(self) -> None:
        # Add row labels (1–8) on the left and right
        for row in range(ROWS):
            # Left-side row label
            row_label = QLabel(str(ROWS - row))
            row_label.setAlignment(Qt.AlignCenter)
            row_label.setFont(QFont("Arial", 12))
            row_label.setStyleSheet("color: black; border: 1px solid black; ")
            if (ROWS - row) % 2 == 0:
                row_label.setStyleSheet("background-color: white;")
            else:
                row_label.setStyleSheet("background-color: grey;")

            row_label.setFixedSize(50, 100)
            self.board_layout.addWidget(row_label, row + 1, 0)

            # Right-side row label
            row_label_clone = QLabel(str(ROWS - row))
            row_label_clone.setAlignment(Qt.AlignCenter)
            row_label_clone.setFont(QFont("Arial", 12))
            row_label_clone.setStyleSheet("color: black; border: 1px solid black;")
            if (ROWS - row) % 2 == 0:
                row_label_clone.setStyleSheet("background-color: grey;")
            else:
                row_label_clone.setStyleSheet("background-color: white;")

            row_label_clone.setFixedSize(50, 100)
            self.board_layout.addWidget(row_label_clone, row + 1, COLUMNS + 1)

        # Add column labels (A–H) on the top and bottom
        for col in range(COLUMNS):
            # Top column label
            col_label = QLabel(chr(ord('A') + col))
            col_label.setAlignment(Qt.AlignCenter)
            col_label.setFont(QFont("Arial", 12))
            col_label.setStyleSheet("color: black; border: 1px solid black;")
            if (COLUMNS - col) % 2 == 0:
                col_label.setStyleSheet("background-color: white;")
            else:
                col_label.setStyleSheet("background-color: grey;")

            col_label.setFixedSize(100, 50)
            self.board_layout.addWidget(col_label, 0, col + 1)

            # Bottom column label
            col_label_clone = QLabel(chr(ord('A') + col))
            col_label_clone.setAlignment(Qt.AlignCenter)
            col_label_clone.setFont(QFont("Arial", 12))
            col_label_clone.setStyleSheet("color: black; border: 1px solid black;")
            if (COLUMNS - col) % 2 == 0:
                col_label_clone.setStyleSheet("background-color: grey;")
            else:
                col_label_clone.setStyleSheet("background-color: white;")

            col_label_clone.setFixedSize(100, 50)
            self.board_layout.addWidget(col_label_clone, ROWS + 1, col + 1)

    def move_successful(self) -> None:
        self._engine.promote_checker()
        self.update_buttons()
        current_colour: str = self._engine.turn_checker()
        self.turn_label.setText(f"{current_colour.capitalize()}'s Turn")  # Update the turn label
        self._engine.check_for_check(current_colour)

    def move_failed(self, message: str) -> None:
        self.update_buttons()
        QMessageBox.critical(self, "Move Failed", f"{message}")

    def get_promotion_choice(self, colour: str, position: tuple[int, int]) -> None:
        dialog = QMessageBox()
        dialog.setWindowTitle("Pawn Promotion")
        # row, col = convert_position(position)
        dialog.setText(f"{colour.capitalize()} pawn can be promoted. Choose a piece:")
        dialog.setIcon(QMessageBox.Question)

        # Add buttons for each piece
        queen_button = dialog.addButton("Queen", QMessageBox.AcceptRole)
        rook_button = dialog.addButton("Rook", QMessageBox.AcceptRole)
        bishop_button = dialog.addButton("Bishop", QMessageBox.AcceptRole)
        knight_button = dialog.addButton("Knight", QMessageBox.AcceptRole)

        # Execute the dialog and determine the choice
        dialog.exec_()

        # Determine which button was clicked
        if dialog.clickedButton() == queen_button:
            choice = 1
        elif dialog.clickedButton() == rook_button:
            choice = 2
        elif dialog.clickedButton() == bishop_button:
            choice = 3
        elif dialog.clickedButton() == knight_button:
            choice = 4
        else:
            choice = 1  # Default to Queen

        # Emit the promotion signal
        self._engine.promote_pawn(choice, position, colour)

    def king_in_check(self, message: str) -> None:
        QMessageBox.critical(self, "King in Check", f"{message}")

    def game_over(self, message: str) -> None:
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Game Over")
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)

        # Add custom buttons
        play_again = msg_box.addButton("Play Again", QMessageBox.YesRole)
        quit_game = msg_box.addButton("Quit", QMessageBox.NoRole)

        # Execute the dialog
        msg_box.exec_()

        # Handle the player's choice
        if msg_box.clickedButton() == play_again:
            print("Starting a new game...")
            self._engine.setup_game()
            # self.game_setup.emit()
            self.update_buttons()
        elif msg_box.clickedButton() == quit_game:
            print("Exiting the game...")
            self.close()

    def make_board_buttons(self) -> None:
        for row in range(ROWS):
            row_buttons: list[QPushButton] = []
            for col in range(COLUMNS):
                button = QPushButton()
                row_buttons.append(button)
                button.setFixedSize(100, 100)
                button.clicked.connect(lambda _, x=row, y=col: self.handle_board_buttons(x, y))
                self.board_layout.addWidget(button, row + 1, col + 1)

            self.board_buttons.append(row_buttons)

    def update_buttons(self) -> None:
        for row, buttons in enumerate(self.board_buttons):
            for col, button in enumerate(buttons):
                if (row + col) % 2 == 0:
                    button.setStyleSheet("background-color: white;")
                else:
                    button.setStyleSheet("background-color: gray;")

                piece: BasePiece | None = self._engine.get_piece(row, col)
                if piece:
                    icon_path = self.icons.get(str(piece), "")
                    if icon_path:
                        button.setIcon(QIcon(icon_path))
                        button.setIconSize(QSize(80, 80))

                else:
                    button.setIcon(QIcon())

    def handle_board_buttons(self, row, col) -> None:
        if self.first_button_press is None:
            self.first_button_press = (row, col)
            piece: BasePiece | None = self._engine.get_piece(row, col)
            if piece is None:
                row, col = convert_position(self.first_button_press)
                QMessageBox.critical(self, "Move Failed", f"No piece at ({row}, {col}).")
                self.first_button_press = None
                return

            self.show_selected_piece(self.first_button_press)
            if piece and piece.colour == self._engine.turn_checker():
                valid_moves: list[tuple[int, int]] = self._engine.get_valid_moves(piece.position)
                self.show_hints(valid_moves)
        else:
            end_position: tuple[int, int] = (row, col)
            self._engine.attempt_move(self.first_button_press, end_position)
            self.first_button_press = None

    def show_selected_piece(self, position: tuple[int, int]) -> None:
        button = self.board_buttons[position[0]][position[1]]
        button.setStyleSheet("background-color: lightblue;")

    def show_hints(self, valid_moves: list[tuple[int, int]]) -> None:
        for row, col in valid_moves:
            button = self.board_buttons[row][col]
            button.setStyleSheet("background-color: lightgreen;")

    def restart_game(self) -> None:
        self._engine.setup_game()
        self.turn_label.setText("White's Turn")  # Update the turn label
        self.update_buttons()

    def white_resigned(self) -> None:
        self.game_over("White as resigned.")

    def black_resigned(self) -> None:
        self.game_over("Black has resigned.")



class Engine(QObject):
    suppress_errors = False

    # Signals
    checkmate = pyqtSignal(str)
    stalemate = pyqtSignal(str)
    insufficient = pyqtSignal(str)
    invalid_move = pyqtSignal(str)
    move_successful = pyqtSignal(tuple, tuple)
    check_warning = pyqtSignal(str)
    promote_pawn_signal = pyqtSignal(str, tuple)

    def __init__(self) -> None:
        super().__init__()
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

    def promote_pawn(self, choice: int, position: tuple[int, int], colour: str) -> None:
        row, col = position
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

    def promote_checker(self) -> None:
        for pieces in (self.board[0], self.board[7]):
            for piece in pieces:
                if piece and isinstance(piece, Pawn):
                    row, col = piece.position
                    if (piece.colour == "white" and row == 0) or (piece.colour == "black" and row == 7):
                        self.promote_pawn_signal.emit(piece.colour, piece.position)

    def castling_checker(self, colour: str, target_position: tuple[int, int]) -> bool:
        king_position: tuple[int, int] = self.king_positions[colour]
        king: BasePiece | None = self.get_piece(king_position[0], king_position[1])
        row: int = king_position[0]
        target_col: int = target_position[1]
        king_col: int = king_position[1]
        if target_col > king_col:  # King side castling
            rook_position: tuple[int, int] = (row, 7)
            rook_col: int = 7
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
        # self._move_piece(rook_position, (row, rook_target_col))
        # self._move_piece(king_position, target_position)
        return True

    def en_passant_test(self, start_position: tuple[int, int], target_position: tuple[int, int]) -> bool:
        delta_row: int = target_position[0] - start_position[0]
        delta_col: int = abs(target_position[1] - start_position[1])
        piece: BasePiece | None = self.get_piece(start_position[0], start_position[1])
        # Should be a pawn at (start_row, end_col)
        target_piece: BasePiece | None = self.get_piece(start_position[0], target_position[1])
        if not isinstance(piece, Pawn) or not isinstance(target_piece, Pawn):
            # Can only do the En-passant move with 2 pawns
            return False

        direction: int = 1 if piece.colour == "black" else -1  # Pawns can only more forwards
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
                if delta_row == direction and delta_col == 1:
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

    def check_for_check(self, king_colour: str) -> None:
        if self.is_checkmate():
            return

        if self.is_stalemate():
            self.stalemate.emit("Game drawn by stalemate...")
            return

        if self.is_insufficient_pieces():
            self.insufficient.emit("Game drawn by insufficient pieces to form a checkmate.")
            return

        if self.is_king_in_check(king_colour):
            self.check_warning.emit(f"{king_colour.capitalize()}'s King is in check.")

    def is_king_in_check(self, king_colour: str):
        king_position: tuple[int, int] = self.king_positions[king_colour]
        for row in self.board:
            for piece in row:
                if piece and piece.colour != king_colour:
                    if piece.check_move(self.board, king_position) and piece.capture_test(self.board, king_position):
                        return True

        return False

    def attempt_move(self, start_position: tuple[int, int], target_position: tuple[int, int]) -> None:
        move: int = self.validate_move(start_position, target_position)
        # Negative messages should have already been sent to view controller
        if move == 0:
            return
        elif move == 1:
            self._move_piece(start_position, target_position)
        elif move == 2:
            self.en_passant_move(start_position, target_position)
        elif move == 3:
            if target_position[1] > start_position[1]:
                rook_col: int = 5
                rook_position: tuple[int, int] = self.get_piece(start_position[0], 7).position
            else:
                rook_col: int = 3
                rook_position: tuple[int, int] = self.get_piece(start_position[0], 0).position

            self._move_piece(rook_position, (start_position[0], rook_col))
            self._move_piece(start_position, target_position)

        self.move_successful.emit(start_position, target_position)

    def validate_move(self, start_position: tuple[int, int], target_position: tuple[int, int]) -> int:
        # have this return an int. 0 = fail, 1 = normal move, 2 = en-passant, 3 = castling
        start_row, start_col = start_position
        target_row, target_col = target_position
        piece: BasePiece | None = self.board[start_row][start_col]
        target_piece: BasePiece | None = self.board[target_row][target_col]
        if piece is None:
            self.emit_invalid_move_signal("Need to select a piece to move.")
            # self.invalid_move.emit("Need to select a piece to move.")
            return 0

        if piece.colour != self.turn_checker():
            self.emit_invalid_move_signal(f"Not {piece.colour}'s turn.")
            # self.invalid_move.emit(f"Not {piece.colour}'s turn.")
            return 0

        if isinstance(target_piece, King):
            self.emit_invalid_move_signal("Can't capture a King")
            # self.invalid_move.emit("Can't capture a King")
            return 0

        if isinstance(piece, King):
            delta_col = abs(target_col - start_col)
            if delta_col == 2 and self.castling_checker(piece.colour, target_position):
                return 3

        if isinstance(piece, Pawn):
            # In here need to do the en-passant stuff as well as other pawn related stuff
            delta_col = abs(target_col - start_col)
            # Normal Pawn move
            if delta_col == 0 and piece.check_move(self.board, target_position):
                if not self.check_test(start_position, target_position):
                    # Pawn check move function will have already allowed a normal move
                    # self._move_piece(start_position, target_position)
                    return 1

            # Normal Pawn capture
            if piece.check_move(self.board, target_position) and not self.check_test(start_position, target_position):
                # self._move_piece(start_position, target_position)
                return 1

            # En-Passant Pawn capture
            if self.en_passant_test(start_position, target_position):
                pieces: tuple[BasePiece, BasePiece] = piece, self.get_piece(start_row, target_col)
                if not self.en_passant_king_test(pieces, target_position):
                    # self.en_passant_move(start_position, target_position)
                    return 2

            # Anything else is bad :(
            self.emit_invalid_move_signal(f"{convert_position(start_position)} -> "
                                          f"{convert_position(target_position)} is an invalid move.")

            # self.invalid_move.emit(f"{convert_position(start_position)} -> "
            #                        f"{convert_position(target_position)} is an invalid move.")
            return 0

        if not piece.check_move(self.board, target_position):
            self.emit_invalid_move_signal(f"{convert_position(start_position)} -> "
                                          f"{convert_position(target_position)} is an invalid move.")

            # self.invalid_move.emit(f"{convert_position(start_position)} -> "
            #                        f"{convert_position(target_position)} is an invalid move.")
            return 0

        if not piece.capture_test(self.board, target_position):
            self.emit_invalid_move_signal("Can't capture that piece.")
            # self.invalid_move.emit("Can't capture that piece")
            return 0

        if self.check_test(start_position, target_position):
            self.emit_invalid_move_signal("Move would put your king in check, try another.")
            # self.invalid_move.emit("Move would put your king in check. Try another.")
            return 0

        # All checks passed, thus move the piece and return true
        # self._move_piece(start_position, target_position)
        return 1

    def get_valid_moves(self, position: tuple[int, int]) -> list[tuple[int, int]]:
        valid_moves: list[tuple[int, int]] = []
        start_row, start_col = position
        piece: BasePiece | None = self.get_piece(start_row, start_col)
        if piece is None:
            return valid_moves

        with suppress_errors(self):  # Suppress the invalid move popups
            for row, squares in enumerate(self.board):
                for col, square in enumerate(squares):
                    if (row, col) != (start_row, start_col) and self.validate_move(position, (row, col)):
                        valid_moves.append((row, col))

        return valid_moves

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
            if (can_block or can_capture or can_escape) and len(threats) == 1:
                continue

            # Unable to be blocked or piece captured -> King is checkmated
            self.game_over.emit(f"{colour.capitalize()}'s King is checkmated. Game Over.")
            return True

        # Either no threats found of preventable threats found -> not checkmate
        return False

    def is_stalemate(self) -> bool:
        colour: str = self.turn_checker()
        valid_moves: list[tuple[int, int]] = []
        for row in self.board:
            for piece in row:
                if piece and piece.colour == colour:
                    valid_moves.extend(self.get_valid_moves(piece.position))

        return True if len(valid_moves) == 0 else False

    def is_insufficient_pieces(self) -> bool:
        pieces: dict[str: list[BasePiece]] = {"white": [], "black": []}
        for row in self.board:
            for piece in row:
                if piece:
                    pieces[piece.colour].append(piece)

        # Special case: King + Bishop vs King + Bishop on same color squares
        if len(pieces["white"]) == 2 and len(pieces["black"]) == 2:
            white_bishop = isinstance(pieces["white"][1], Bishop)
            black_bishop = isinstance(pieces["black"][1], Bishop)
            if white_bishop and black_bishop:
                # Check if the bishops are on the same color
                white_square_color = sum(pieces["white"][1].position) % 2
                black_square_color = sum(pieces["black"][1].position) % 2
                if white_square_color == black_square_color:
                    return True

                # Checkmate is possible if the bishops are on opposite square colours
                return False

        white_min = is_minimal(pieces["white"])
        black_min = is_minimal(pieces["black"])
        return white_min and black_min

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
        if len(threats) > 1:
            return False

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

    def emit_invalid_move_signal(self, message: str) -> None:
        if not self.suppress_errors:
            self.invalid_move.emit(message)


class BasePiece:
    def __init__(self, position: tuple[int, int], colour: str) -> None:
        self.position = position
        self.colour = colour
        self.never_moved = True

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


@contextmanager
def suppress_errors(engine: Engine):
    engine.suppress_errors = True  # Temporarily suppress errors
    try:
        yield  # Run the block of code
    finally:
        engine.suppress_errors = False  # Reset suppression


def out_of_bounds(position: tuple[int, int]) -> bool:
    row, col = position
    return row < 0 or row > 7 or col < 0 or col > 7


def convert_position(position: tuple[int, int]) -> tuple[int, str]:
    row, col = position
    row = 8 - row
    col = chr(col + ord("A"))
    return row, col


def is_minimal(pieces_list: list[BasePiece]) -> bool:
    if len(pieces_list) == 1:  # Only the king remains
        return True
    if len(pieces_list) == 2:  # King and either a bishop or a knight
        return isinstance(pieces_list[1], (Bishop, Knight))

    # All other situations are fine
    return False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    controller = ViewController()
    controller.show()
    sys.exit(app.exec_())
