from pgl import GWindow, GOval, GRect, GCompound, GLabel, GLine
from sudoku import SudokuBoard

# Constants

GWINDOW_WIDTH = 480               # Width of the graphics window
MENU_BAR_HEIGHT = 60
GWINDOW_HEIGHT = GWINDOW_WIDTH + MENU_BAR_HEIGHT   # Height of the graphics window
MENU_BAR_BGCOLOR = "gray"
MENU_TITLE_FONT = "36px 'Helvetica Neue','Arial','Sans-Serif'"
MENU_TITLE_COLOR = "white"
BUTTON_RADIUS = MENU_BAR_HEIGHT//8


CELL_WIDTH = GWINDOW_WIDTH // 4
CELL_GOOD_COLOR = "#CFF0CC"
CELL_BAD_COLOR = "pink"
CELL_BORDER_COLOR = "black"
CELL_FONT = "72px 'Helvetica Neue','Arial','Sans-Serif'"
CELL_TEXT_COLOR = "black"
SUGGESTION_TEXT_COLOR = "#cccccc"
SUBCELL_FILL_COLOR = "light blue"
SUBCELL_FONT = "36px 'Helvetica Neue','Arial','Sans-Serif'"
SUBCELL_TEXT_COLOR = "#ffffff"

BOX_WIDTH = CELL_WIDTH * 2
SUBCELL_WIDTH = CELL_WIDTH // 2



class SudokuDigitSelectorSubcell(GCompound):
    
    def __init__(self, digit):
        GCompound.__init__(self)
        self.digit = str(digit)
        cell = GRect(0, 0, SUBCELL_WIDTH, SUBCELL_WIDTH)
        cell.setColor(CELL_BORDER_COLOR)
        cell.setFillColor(SUBCELL_FILL_COLOR)
        cell.setFilled(True)
        self.add(cell, 0, 0)
        self.label = GLabel(digit)
        self.label.setFont(SUBCELL_FONT)
        self.label.setColor(SUBCELL_TEXT_COLOR)
        self.add(self.label, 
                 SUBCELL_WIDTH//2 - self.label.getWidth()//2, 
                 SUBCELL_WIDTH//2 + self.label.getAscent()//2 - 3)

class ClearCell(GCompound):
     def __init__(self):
        GCompound.__init__(self)
        self.digit = ""
        outer = GOval(-20, -20, 40, 40)
        outer.setFilled(True)
        outer.setFillColor("red")
        outer.setColor("red")
        self.add(outer)
        outer = GOval(-16, -16, 32, 32)
        outer.setFilled(True)
        outer.setFillColor("white")
        outer.setColor("white")
        self.add(outer)
        strikethrough = GLine(15, -15, -15, 15)
        strikethrough.setLineWidth(6)
        strikethrough.setColor("red")
        self.add(strikethrough)

        
class SudokuDigitSelector(GCompound):

    def __init__(self):
        GCompound.__init__(self)
        self.add(SudokuDigitSelectorSubcell(1), 0, 0)
        self.add(SudokuDigitSelectorSubcell(2), SUBCELL_WIDTH, 0)
        self.add(SudokuDigitSelectorSubcell(3), 0, SUBCELL_WIDTH)
        self.add(SudokuDigitSelectorSubcell(4), SUBCELL_WIDTH, SUBCELL_WIDTH)
        self.add(ClearCell(), SUBCELL_WIDTH, SUBCELL_WIDTH)

    def mouseup(self, x, y):
        if self.getElementAt(x, y) is not None:
            return self.getElementAt(x, y).digit
        else:
            return None


class SudokuCell(GCompound):
    
    def __init__(self, digit):
        GCompound.__init__(self)
        if digit != 0:
            self.digit = str(digit)
        else:
            self.digit = None
        self.cell = GRect(0, 0, CELL_WIDTH, CELL_WIDTH)
        self.cell.setColor(CELL_BORDER_COLOR)
        self.cell.setFillColor(CELL_GOOD_COLOR)
        self.cell.setFilled(True)        
        self.add(self.cell, 0, 0)  
        self.label = None
        self.only_a_suggestion = True
        self.render_label()
        self.selector = None
        
    def get_digit(self):
        if self.digit == None or self.digit == '' or self.only_a_suggestion:
            return 0
        else:
            return int(self.digit)

    def set_background_color(self, color):
        self.cell.setFillColor(color)

    def suggest_solution(self, digit):        
        if self.only_a_suggestion:
            self.digit = str(digit)
            self.render_label()

    def render_label(self):
        if self.label is not None:
            self.remove(self.label)
        if self.digit is not None and self.digit != "0":
            self.label = GLabel(self.digit)
        else:
            self.label = GLabel("")
        self.label.setFont(CELL_FONT)
        if self.only_a_suggestion:
            self.label.setColor(SUGGESTION_TEXT_COLOR)
        else:            
            self.label.setColor(CELL_TEXT_COLOR)
        self.add(self.label, 
                 CELL_WIDTH//2 - self.label.getWidth()//2, 
                 CELL_WIDTH//2 + self.label.getAscent()//2 - 7)

        
    def mousedown(self):
        self.selector = SudokuDigitSelector()
        self.add(self.selector, 0, 0)        

    def mouseup(self, x, y):
        if self.selector is not None:
            digit = self.selector.mouseup(x, y)            
            if digit is not None:
                self.digit = digit
                if str(self.digit) == "0" or self.digit == "":
                    self.digit = ""
                    self.only_a_suggestion = True
                elif self.digit != "":
                    self.only_a_suggestion = False    
                self.render_label()
            self.remove(self.selector)
            self.selector = None

 
class SudokuBox(GCompound):

    def __init__(self, cells):
        GCompound.__init__(self)
        assert len(cells) == 4
        self.cells = cells
        self.add(cells[0], 0, 0)
        self.add(cells[1], CELL_WIDTH, 0)
        self.add(cells[2], 0, CELL_WIDTH)
        self.add(cells[3], CELL_WIDTH, CELL_WIDTH)

    def mousedown(self, x, y):
        self.getElementAt(x, y).mousedown()

    def mouseup(self, x, y):
        for cell in self.cells:
            cell.mouseup(x % CELL_WIDTH, y % CELL_WIDTH)

    def set_background_color(self, color):
        for cell in self.cells:
            cell.set_background_color(color)
            
    def as_matrix(self):
        return [[self.cells[0].get_digit(), self.cells[1].get_digit()],
                [self.cells[2].get_digit(), self.cells[3].get_digit()]]
                
    def suggest_solution(self, solution):
        self.cells[0].suggest_solution(solution[0][0])
        self.cells[1].suggest_solution(solution[0][1])
        self.cells[2].suggest_solution(solution[1][0])
        self.cells[3].suggest_solution(solution[1][1])

        
class VisualSudokuBoard(GCompound):

    def __init__(self, boxes):
        GCompound.__init__(self)
        self.boxes = boxes
        assert len(boxes) == 4
        self.add(boxes[0], 0, 0)
        self.add(boxes[1], BOX_WIDTH, 0)
        self.add(boxes[2], 0, BOX_WIDTH)
        self.add(boxes[3], BOX_WIDTH, BOX_WIDTH)

    def mousedown(self, x, y):
        self.getElementAt(x, y).mousedown(x % BOX_WIDTH, y % BOX_WIDTH)

    def mouseup(self, x, y):
        for box in self.boxes:
            box.mouseup(x % BOX_WIDTH, y % BOX_WIDTH)
        self.check_satisfiability()

    def check_satisfiability(self):        
        board = self.get_board()
        solution = board.solve()
        if solution is None:
            self.suggest_solution([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
            self.set_background_color(CELL_BAD_COLOR)
        else:
            self.suggest_solution(solution.matrix)
            self.set_background_color(CELL_GOOD_COLOR)
            
    def set_background_color(self, color):
        for box in self.boxes:
            box.set_background_color(color)
            
    def suggest_solution(self, solution):
        box1 = [[solution[0][0], solution[0][1]],
                [solution[1][0], solution[1][1]]]
        box2 = [[solution[0][2], solution[0][3]],
                [solution[1][2], solution[1][3]]]
        box3 = [[solution[2][0], solution[2][1]],
                [solution[3][0], solution[3][1]]]
        box4 = [[solution[2][2], solution[2][3]],
                [solution[3][2], solution[3][3]]]
        self.boxes[0].suggest_solution(box1)
        self.boxes[1].suggest_solution(box2)
        self.boxes[2].suggest_solution(box3)
        self.boxes[3].suggest_solution(box4)
            
    def get_board(self):
        matrices = [box.as_matrix() for box in self.boxes]
        row1 = matrices[0][0] + matrices[1][0]
        row2 = matrices[0][1] + matrices[1][1]
        row3 = matrices[2][0] + matrices[3][0]
        row4 = matrices[2][1] + matrices[3][1]
        return SudokuBoard([row1, row2, row3, row4])
        

class MenuBar(GCompound):
    def __init__(self):
        GCompound.__init__(self)
        bar = GRect(GWINDOW_WIDTH, MENU_BAR_HEIGHT)
        bar.setFilled(True)
        bar.setColor(CELL_BORDER_COLOR)
        bar.setFillColor(MENU_BAR_BGCOLOR)
        self.add(bar, 0, 0)
        self.label = GLabel("Y O U D O K U")
        self.label.setFont(MENU_TITLE_FONT)
        self.label.setColor(MENU_TITLE_COLOR)
        self.add(self.label, 
                 GWINDOW_WIDTH//2 - self.label.getWidth()//2, 
                 MENU_BAR_HEIGHT//2 + self.label.getAscent()//2 - 5)
    
    def mousedown(self, x, y):
        pass

    def mouseup(self, x, y):
        pass

class YoudokuBoard(GCompound):
    def __init__(self):
        GCompound.__init__(self)
        box1 = construct_box(0,0,0,0)
        box2 = construct_box(0,0,0,0)
        box3 = construct_box(0,0,0,0)
        box4 = construct_box(0,0,0,0)
        self.board = VisualSudokuBoard([box1, box2, box3, box4])
        self.add(self.board, 0, 0)
        self.menu = MenuBar()
        self.add(self.menu, 0, GWINDOW_WIDTH)
        self.most_recent_mousedown = None

    def mousedown(self, x, y):
        element = self.getElementAt(x, y)
        if element == self.board:
            element.mousedown(x, y)
        elif element == self.menu:
            element.mousedown(x, y - GWINDOW_WIDTH)
        self.most_recent_mousedown = element

    def mouseup(self, x, y):
        self.most_recent_mousedown.mouseup(x, y)
        self.most_recent_mousedown = None

def construct_box(digit1, digit2, digit3, digit4):
    cell11 = SudokuCell(digit1)
    cell12 = SudokuCell(digit2)
    cell21 = SudokuCell(digit3)
    cell22 = SudokuCell(digit4)
    box = SudokuBox([cell11, cell12, cell21, cell22])
    return box


    

def sudoku_builder():

    def mousedown_action(e):
        element = gw.getElementAt(e.getX(), e.getY())
        if element is not None:
            element.mousedown(e.getX(), e.getY())

    def mouseup_action(e):
        element = gw.getElementAt(e.getX(), e.getY())
        if element is not None:
            element.mouseup(e.getX(), e.getY())
        else:
            board.mouseup(e.getX(), e.getY())

    gw = GWindow(GWINDOW_WIDTH, GWINDOW_HEIGHT)
    gw.addEventListener("mousedown", mousedown_action)
    gw.addEventListener("mouseup", mouseup_action)
    board = YoudokuBoard()
    gw.add(board, 0, 0)



    
if __name__ == "__main__":
    sudoku_builder()