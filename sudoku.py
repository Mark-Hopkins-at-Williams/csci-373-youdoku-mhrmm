import math
import cnf
import copy
from dpll import dpll
from search import search_solver


def irange(i, j):
    return range(i, j+1)


def lit(d, i, j, polarity=True):
    """
    Creates a literal indicating whether cell (i, j) contains digit d.
    
    """
    literal = 'd{}_{}_{}'.format(d, i, j)
    if not polarity:
        literal = '!{}'.format(literal)
    return literal


class SudokuBoard:
    """Representation of a Sudoku board."""
    
    def __init__(self, matrix):
        """
        Parameters
        ----------
        matrix : list[list[int]]
            A two-dimensional array, providing the initial values of each cell.
            Zero represents an empty cell.
        """

        self.matrix = matrix
        self.box_width = int(math.sqrt(len(self.matrix)))
        err = 'Improper dimensions for a Sudoku board!'
        assert self.box_width == math.sqrt(len(self.matrix)), err
        self.generic_clauses = list(set(self._generate_generic_clauses()))

    def __str__(self):
        """A simple string representation of the board.

        See TestSudokuBoard.test_board_str and TestSudokuBoard.test_board_str2
        in test.py for examples of expected output.
        """

        row_strs = [''.join([str(digit) for digit in row]) for row in self.matrix]        
        return '\n'.join(row_strs)
 
    def rows(self):
        """Returns the row addresses of the board.

        Specifically, this returns a list of sets, where each set ccorresponds to
        the addresses of a single row. For a 2x2 Sudoku board, this would be:

        [{(1, 1), (1, 2), (1, 3), (1, 4)},
         {(2, 1), (2, 2), (2, 3), (2, 4)},
         {(3, 1), (3, 2), (3, 3), (3, 4)},
         {(4, 1), (4, 2), (4, 3), (4, 4)}]

        The order of the rows in the list should be top-to-bottom.
        """

        def row_cells(i, row_length):
            return set([(i, j) for j in irange(1, row_length)])
        num_symbols = self.box_width * self.box_width
        return [row_cells(row, num_symbols) for row in irange(1, num_symbols)]
    
    def columns(self):
        """Returns the column addresses of the board.

        Specifically, this returns a list of sets, where each set ccorresponds to
        the addresses of a single column. For a 2x2 Sudoku board, this would be:

        [{(1, 1), (2, 1), (3, 1), (4, 1)},
         {(1, 2), (2, 2), (3, 2), (4, 2)},
         {(1, 3), (2, 3), (3, 3), (4, 3)},
         {(1, 4), (2, 4), (3, 4), (4, 4)}]

        The order of the columns in the list should be left-to-right.
        """

        def col_cells(j, col_length):
            return set([(i, j) for i in irange(1, col_length)])
        num_symbols = self.box_width * self.box_width
        return [col_cells(col, num_symbols) for col in irange(1, num_symbols)]
 
    def boxes(self):
        """Returns the addresses of each box of the board.

        Specifically, this returns a list of sets, where each set ccorresponds to
        the addresses of a single box. For a 2x2 Sudoku board, this would be:

        [{(1, 1), (1, 2), (2, 1), (2, 2)},
         {(1, 3), (1, 4), (2, 3), (2, 4)},
         {(3, 1), (3, 2), (4, 1), (4, 2)},
         {(3, 3), (3, 4), (4, 3), (4, 4)}]

        The order of the columns in the list should be left-to-right, then
        top-to-bottom.
        """

        def box_cells(a, b, box_width):
            return set([(i+1,j+1) for i in range((a-1) * box_width, a * box_width)
                                  for j in range((b-1) * box_width, b * box_width)])
        return [box_cells(a, b, self.box_width) for a in irange(1, self.box_width)
                                                 for b in irange(1, self.box_width)]

    def zones(self):
        return self.rows() + self.columns() + self.boxes()
    
    def contents(self):
        """Computes a set of clauses that describe the current board state.

        In other words, this creates clauses that describe the numbers have already
        been filled in. For instance, if the board were:

        board = SudokuBoard([ [0, 0, 0, 3],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 1, 0, 0] ])

        then ```board.contents()``` should return a set of Clauses equivalent to:

        { cnf.c('d3_1_4'), cnf.c('d1_4_2') }

        The first clause (```cnf.c('d3_1_4')```) asserts that the digit 3 must appear
        at address (1,4), whereas the second clause asserts that the digit 1 must
        appear at address (4,2).

        Returns
        -------
        set[Clause]
            the set of clauses that describe the current board state
        """

        clauses = []
        for row in range(len(self.matrix)):
            for col in range(len(self.matrix[row])):
                digit = self.matrix[row][col]
                if digit != 0:
                    clause = cnf.c(lit(digit, row+1, col+1))
                    clauses.append(clause)
        return clauses
    
    def cnf(self):
        """Constructs a cnf.Cnf instance that fully describes this SudokuBoard.

        Note that the CNF sentence should express both the rules of Sudoku (each zone
        contains exactly one of each digit, no cell is empty) and the current board
        state (which cells have already been filled in by particular digits).

        Returns
        -------
        Cnf
            a CNF sentence describing this sudoku board
        """

        clauses = self.generic_clauses + self.contents()
        return cnf.Cnf(clauses)

    def solve(self):
        """Constructs a new Sudokuboard corresponding to a valid puzzle completion.

        For instance, if

            board = SudokuBoard([[4, 1, 2, 3],
                                 [3, 4, 1, 2],
                                 [2, 3, 4, 1],
                                 [0, 0, 0, 0]])

        Then board.solve() should return a new SudokuBoard instance equivalent to:

            SudokuBoard([[4, 1, 2, 3],
                         [3, 4, 1, 2],
                         [2, 3, 4, 1],
                         [1, 2, 3, 4]])

        If there are no valid completions, then this should return None.
        If there are multiple valid completions, then any may be returned.
        """
        def interpret_lit(l):
            negate = (l[0] == "!")
            if negate:
                l = l[2:]
            else:
                l = l[1:]
            d, i, j = l.split("_")
            return d, i, j, negate
        model = dpll(self.cnf())
        if model is None:
            return None
        matrix = copy.deepcopy(self.matrix)
        positive_literals = [l for l in model if model[l] == 1]
        for l in positive_literals:
            d, i, j, _ = interpret_lit(l)
            matrix[int(i)-1][int(j)-1] = int(d)
        return SudokuBoard(matrix)

    def _generate_generic_clauses(self):
        num_symbols = self.box_width * self.box_width
        clause_strs = []
        for zone in self.zones():
            for digit in irange(1, num_symbols):
                clause_strs += exactly_one_clauses(zone, digit)
        clause_strs += nonempty_clauses(self.box_width)
        clauses = [cnf.c(clause) for clause in clause_strs]
        return clauses
    
def at_least_clause(zone, d):
    """Creates clauses for the constraint "this zone must contain digit d at least once".

    Specificially, this takes a set `zone` of cell addresses and a digit `d`.
    It should produce a string representation of the clause corresponding to the
    constraint "digit `d` should appear at least once among the addresses in
    `zone`". For instance:

        at_least_clause({(1, 3), (1, 4), (2, 3), (2, 4)}, d=2)

    should return the string:

        'd2_1_3 || d2_1_4 || d2_2_3 || d2_2_4'

    For this string, the literals are expected to be listed in alphabetical
    order (according to a string comparison).

    Parameters
    ----------
    zone : set[tuple[int]]
        a set of cell addresses
    d : int
        a digit of the Sudoku puzzle

    Returns
    -------
    str
        a string representation of the clause corresponding to the constraint
        "this zone must contain digit d at least once"
    """

    literals = [lit(d, i, j) for (i, j) in sorted(zone)]
    return ' || '.join(literals)  


def at_most_clauses(cells, d):
    """Creates clauses for the constraint "this zone must contain digit d at most once".

    Specificially, this takes a set `zone` of cell addresses and a digit `d`.
    It should produce a string representation of the clause corresponding to the
    constraint "digit `d` should appear at most once among the addresses in
    `zone`". For instance:

        at_most_clauses({(1, 3), (1, 4), (2, 3), (2, 4)}, d=2)

    should return the list:

        ['!d2_1_3 || !d2_1_4',
         '!d2_1_3 || !d2_2_3',
         '!d2_1_3 || !d2_2_4',
         '!d2_1_4 || !d2_2_3',
         '!d2_1_4 || !d2_2_4',
         '!d2_2_3 || !d2_2_4']

    For this string, the literals are expected to be listed in alphabetical
    order (according to a string comparison).

    Parameters
    ----------
    zone : set[tuple[int]]
        a set of cell addresses
    d : int
        a digit of the Sudoku puzzle

    Returns
    -------
    str
        a string representation of the clause corresponding to the constraint
        "this zone must contain digit d at most once"
    """

    def all_pairs(seq):
        for a in range(len(seq)):
            for b in range(a+1, len(seq)):
                yield seq[a], seq[b]
    clauses = []         
    for cell1, cell2 in all_pairs(sorted(cells)):
        clauses.append('{} || {}'.format(lit(d, cell1[0], cell1[1], polarity=False), 
                                         lit(d, cell2[0], cell2[1], polarity=False)))        
    return clauses


def exactly_one_clauses(cells, d):
    """
    Encodes: "The following cells have exactly 1 of digit d."

    """
    return [at_least_clause(cells, d)] + at_most_clauses(cells, d)     


def nonempty_clauses(box_width):
    """Creates clauses for the constraint "no cell can be empty".

    Specificially, this takes as argument the width of a box on your Sudoku board.
    It should produce a list of the string representations of the clauses
    corresponding to the constraint "no cell can be empty". For instance,
    the call nonempty_clauses(2)

    should return the list:

        ['d1_1_1 || d2_1_1 || d3_1_1 || d4_1_1',
         'd1_1_2 || d2_1_2 || d3_1_2 || d4_1_2',
         'd1_1_3 || d2_1_3 || d3_1_3 || d4_1_3',
         'd1_1_4 || d2_1_4 || d3_1_4 || d4_1_4',
         'd1_2_1 || d2_2_1 || d3_2_1 || d4_2_1',
         'd1_2_2 || d2_2_2 || d3_2_2 || d4_2_2',
         'd1_2_3 || d2_2_3 || d3_2_3 || d4_2_3',
         'd1_2_4 || d2_2_4 || d3_2_4 || d4_2_4',
         'd1_3_1 || d2_3_1 || d3_3_1 || d4_3_1',
         'd1_3_2 || d2_3_2 || d3_3_2 || d4_3_2',
         'd1_3_3 || d2_3_3 || d3_3_3 || d4_3_3',
         'd1_3_4 || d2_3_4 || d3_3_4 || d4_3_4',
         'd1_4_1 || d2_4_1 || d3_4_1 || d4_4_1',
         'd1_4_2 || d2_4_2 || d3_4_2 || d4_4_2',
         'd1_4_3 || d2_4_3 || d3_4_3 || d4_4_3',
         'd1_4_4 || d2_4_4 || d3_4_4 || d4_4_4']

    For this string, the literals are expected to be listed in alphabetical
    order (according to a string comparison).

    Parameters
    ----------
    box_width : int
        the width of a box of the Sudoku board

    Returns
    -------
    str
        a string representation of the clause corresponding to the constraint
        "no cell can be empty"
    """

    def non_zero_cell(i, j):
        """
        Encodes: "Cell i, j is non-zero."
    
        """ 
        literals = [lit(d, i, j) for d in irange(1, box_size)]
        return ' || '.join(literals)         
    box_size = box_width * box_width    
    clauses = [non_zero_cell(row, col)
               for row in irange(1, box_size)
               for col in irange(1, box_size)]
    return clauses
                
