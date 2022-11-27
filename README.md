![screenshot](images/youdoku_screenshot.png)

## Youdoku

An introduction to the assignment can be found [here](https://docs.google.com/document/d/1w3E3Ecty61IRGf22Wh83vXuEy8GQ6FTuLslPmnis4I0/edit?usp=sharing).
Unlike other homeworks, the main steps you need to follow are contained within this README.

This repository contains starter code:
- `sudoku.py`: Code for part A goes in this file, as well as code for B1 and B6.
- `search.py`: Code for B2 goes in this file.
- `dpll.py`: Code for B3-B5 goes in this file.

It also contains code that you should use and are allowed to extend (but be careful not to break
the unit tests if you do choose to extend these classes):
- `cnf.py`: Basic data structures for manipulating CNF sentences

It also contains code that you should use but SHOULD NOT change:
- `util.py`: Basic tools for search, like the SearchSpace interface and depth-first search.

It also contains the Youdoku executable that you will use but should not change:
- `youdoku.py`: The Youdoku product.

Ignore (but do not delete!!!) the other files in this repository.


## Part A: Expressing Sudoku in CNF

1. In ```sudoku.py```, create a class called SudokuBoard such that we can 
   create a 2x2 sudoku board by typing:
   
           board = SudokuBoard([[0, 0, 0, 3], 
                                [0, 0, 0, 2], 
                                [3, 0, 0, 0], 
                                [4, 0, 0, 0]])

   The digit "0" indicates that the cell has not yet been filled in. In this
   example, cells (1,4) and (3,1) are filled with a "3", cell (2,4) is filled
   with a "2", and cell (4,1) is filled with a "4".
   
   Note that the constructor should accept any kxk board, where k >= 2.
   
   Make sure it supports the following methods:
   
   - ```str(board)``` should return a simple string representation of the 
   board. For the above example, the string representation should 
   be ```"0003\n0002\n3000\n4000"```.
   - ```board.rows()``` should return a list of sets, where each set
   corresponds to the addresses of a single row. For a 2x2 Sudoku board,
   this would be:
   
       [{(1, 1), (1, 2), (1, 3), (1, 4)}, 
        {(2, 1), (2, 2), (2, 3), (2, 4)}, 
        {(3, 1), (3, 2), (3, 3), (3, 4)}, 
        {(4, 1), (4, 2), (4, 3), (4, 4)}]  
        
   The order of the rows in the list should be top-to-bottom.
   - ```board.columns()``` should return a list of sets, where each set
   corresponds to the addresses of a single column. For a 2x2 Sudoku board,
   this would be:
   
       [{(1, 1), (2, 1), (3, 1), (4, 1)}, 
        {(1, 2), (2, 2), (3, 2), (4, 2)}, 
        {(1, 3), (2, 3), (3, 3), (4, 3)}, 
        {(1, 4), (2, 4), (3, 4), (4, 4)}]
                
   Note that the order of the columns in the list should be left-to-right.
   - ```board.boxes()``` should return a list of sets, where each set
   corresponds to the addresses of a single box. For a 2x2 Sudoku board,
   this would be:
   
       [{(1, 1), (1, 2), (2, 1), (2, 2)}, 
        {(1, 3), (1, 4), (2, 3), (2, 4)}, 
        {(3, 1), (3, 2), (4, 1), (4, 2)}, 
        {(3, 3), (3, 4), (4, 3), (4, 4)}]
                
   Note that the order of the boxes in the list should be left-to-right, 
   then top-to-bottom.

   Once you have a successful implementation, the following unit tests should
   succeed:
   
       python -m unittest test.TestSudokuBoard

2. Now we want to create clauses according to our CNF formulation from
   HW: Sudoku 2. First, create clauses for the constraint "each zone must
   contain digit d at least once". In ```sudoku.py```, create a function 
   ```at_least_clause(A, d)``` which takes a set ```A``` of cell addresses
   and a digit ```d```. It should produce a string representation of the 
   clause corresponding to the constraint "digit ```d``` should appear at least
   once among the addresses in ```A```". For instance:
   
       at_least_clause({(1, 3), (1, 4), (2, 3), (2, 4)}, d=2)
       
   should return the string:
   
       'd2_1_3 || d2_1_4 || d2_2_3 || d2_2_4'
       
   For this string, the literals are expected to be listed in alphabetical 
   order (according to a string comparison).

   Once you have successfully implemented the function, the following unit
   tests should succeed:
   
       python -m unittest test.TestAtLeastClause   
   
3. Next, create clauses for the constraint "each zone must
   contain digit d at most once". In ```sudoku.py```, create a function 
   ```at_most_clauses(A, d)``` which takes a set ```A``` of cell addresses
   and a digit ```d```. It should produce a list of the string representations
   of the clauses corresponding to the constraint "digit d should appear at most
   once among the addresses in A". For instance:
   
       at_most_clauses({(1, 3), (1, 4), (2, 3), (2, 4)}, d=2)
       
   should return the list:
   
       ['!d2_1_3 || !d2_1_4', 
        '!d2_1_3 || !d2_2_3', 
        '!d2_1_3 || !d2_2_4', 
        '!d2_1_4 || !d2_2_3', 
        '!d2_1_4 || !d2_2_4', 
        '!d2_2_3 || !d2_2_4']

   Once you have successfully implemented the function, the following unit
   tests should succeed:
   
       python -m unittest test.TestAtMostClauses   
       
4. Create clauses for the constraint "no cell can be empty". 
   In ```sudoku.py```, create a function  ```nonempty_clauses(k)```
   where ```k``` is the width of a box on your Sudoku board.  It should produce 
   a list of the string representations of the clauses corresponding to the 
   constraint "this cell should contain a digit". For instance:
   
       nonempty_clauses(2)
       
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

   Once you have successfully implemented the function, the following unit
   tests should succeed:
   
       python -m unittest test.TestNonemptyClauses   

5. In ```sudoku.SudokuBoard``` create a method called ```contents```
   that computes a set of Clauses that describe the current board state
   (i.e. which numbers have already been filled in). For instance, if
   the board were:
   
       board = SudokuBoard([ [0, 0, 0, 3], 
                             [0, 0, 0, 0], 
                             [0, 0, 0, 0], 
                             [0, 1, 0, 0] ])
       
   then ```board.contents()``` should return a set of Clauses equivalent to:
   
       { cnf.c('d3_1_4'), cnf.c('d1_4_2') }
       
   The first clause (```cnf.c('d3_1_4')```) asserts that the digit 3 must appear
   at address (1,4), whereas the second clause asserts that the digit 1 must
   appear at address (4,2).
   
   In addition, add these clauses to the return value of ```board.cnf()```.
   
   Once you have a successful implementation, the following unit tests should
   succeed:
   
       python -m unittest test.TestContentClauses

6. Finally, put this all together into a method .cnf() of SudokuBoard.
   Calling board.cnf() should construct a cnf.Cnf instance containing all
   the clauses that are needed to express the SudokuBoard board. See the
   unit tests in test.TestCnfConversion for example input/outputs. Note that
   the CNF sentence should express both the rules of Sudoku (each zone contains 
   exactly one of each digit, no cell is empty) and the current board state
   (which cells have already been filled in by particular digits).

   Once you have successfully implemented the method, the following unit
   tests should succeed:
   
       python -m unittest test.TestCnfConversion


## Part B: Search-based Satisfiability

Time to create a SAT solver based on the search strategy discussed in class.

1. We're going to represent a model as a dictionary. Specifically,
   a model that assigns 1 to symbol A, and 0 to symbol B should be represented
   as the following dictionary:
   
       {'A': True, 'B': False}
       
   In this dictionary, use the Boolean value ```True``` for 1 and
   the Boolean value ```False``` for 0.

   Add a method ```.check_model(model)``` to ```cnf.Cnf``` that takes a
   model (represented as a dictionary, described above) and returns
   whether it satisfies the logical sentence. For instance, if we have
   the following CNF sentence:
   
       sent = sentence('a || b', '!a || !b')
       
   Then the following two calls should return ```False```:
   
       sent.check_model({'a': False, 'b': False})
       sent.check_model({'a': True, 'b': True})
       
   But the following two calls should return ```True```:
   
       sent.check_model({'a': False, 'b': True})
       sent.check_model({'a': True, 'b': False})
       
   Once you have a successful implementation, the following unit tests should
   succeed:
   
       python -m unittest test.TestCheckModel

2. Next, complete the implementation of ```SatisfiabilitySearchSpace```
   in ```search.py``` so that we can run depth-first search on it and obtain 
   a satisfying model (as implemented already by the ```search_solver``` function, also
   in ```search.py```). The states of this search space should be tuples of Literals, e.g.

         (Literal("A"), Literal("B"), Literal("!C"), Literal("!D"))

   is a state that has assigned one to signature symbols A and B, and has 
   assigned zero to symbols C and D. The successors of a state should assign the
   next unassigned signature symbol. For instance, if the next signature symbol is
   "E", then calling .get_successors() on the above state should return:

         [(Literal("A"), Literal("B"), Literal("!C"), Literal("!D"), Literal(!E")),
          (Literal("A"), Literal("B"), Literal("!C"), Literal("!D"), Literal(E"))]

   In other words, it extends the current assignment to both possible assignments
   of symbol E. 

   You need to complete two methods:
   - .get_successors(state) behaves as described above
   - .is_goal_state(state) takes a state and returns True if it is a goal state, 
     i.e., all symbols have been assigned, and the corresponding model satisfies
     self.sentence.
   
   Make sure to obey the following conventions:
   - The positive polarity of a symbol should be explored before the negative
     polarity (because dfs uses a stack, this means the negative polarity
     successor should appear first in the result of .get_successors())
   - Symbols should be assigned in alphabetical order.
   
   Once you have a successful implementation, the following unit tests should succeed:
   
         python -m unittest test.TestSearchSolver

3. The bare-bones search is nice in the sense that it eventually gives us a 
   satisfying model. But it's very slow at solving Sudoku puzzles, so let's add unit
   resolution to speed it up. In ```dpll.py```, implement the function ```unit_resolve```. 
   This function resolves a "target" clause simultaneously with a set of
   unit clauses, according to the following procedure:
   - If the target clause contains the same literal as one of the unit clauses,
     e.g. the target clause is !A || !B || C and one the unit clauses is !B, then
     the target clause is redundant (entailed by that unit clause) and therefore
     unnecessary. Hence None should be returned.
   - Otherwise, any target clause literals whose negations appear in a unit clause
     should be removed, e.g. if the target clause is !A || !B || C || !D and the unit
     clauses contain both A and !C, then the resolved clause should be !B || D.

   Once you have a successful implementation, the following unit tests should succeed:
   
     python -m unittest test.TestUnitResolve

4. In ```dpll.py```, implement the function ```unit_resolution```, which
   takes two sets of clauses as arguments: a set of "regular" clauses and a
   set of unit clauses. This function should use the unit_resolve function 
   to resolve each regular clause AND unit clause with the unit clauses.

   Attention: resolution can produce new unit clauses, and these unit clauses
   must also be resolved with all the other clauses. The process should continue
   until no new clauses can be created through unit resolution.

   See the examples in test.TestUnitResolution to gain further insight into
   the expected behavior of this function.

   Once you have a successful implementation, the following unit tests should succeed:
   
     python -m unittest test.TestUnitResolution

5. Now complete the implementation of the DpllSearchSpace class (which
   inherits from SatisfiabilitySearchSpace in ```search.py```) by overriding
   the .get_successors method. As with SatisfiabilitySearchSpace, a search state of 
   the DpllSearchSpace is a tuple of literals, one for each symbol in the signature.
   The successors of state (l_1, ..., l_k) should typically be (l_1, ..., l_k, !s_{k+1}) 
   and (l_1, ..., l_k, s_{k+1}), where s_{k+1} is the (k+1)th symbol in the
   signature (according to an alphabetical ordering of the signature symbols).

   However:
   - if self.sent conjoined with literals l_1, ..., l_k entails False (according
     to unit resolution), then there should be no successors, i.e. this method
     should return an empty list
   - if self.sent conjoined with literals l_1, ..., l_k entails !s_{k+1}
     (according to unit resolution), then the only successor should be
     (l_1, ..., l_k, !s_{k+1})
   - if self.sent conjoined with literals l_1, ..., l_k entails s_{k+1},
     (according to unit resolution), then the only successor should be
     (l_1, ..., l_k, s_{k+1}).

   See the examples in test.TestDpllSearchSpace to gain further insight into
   the expected behavior of this method. Once you have a successful implementation, 
   the following unit tests should succeed:
   
       python -m unittest test.TestDpllSearchSpace
       python -m unittest test.TestDpll

   You should also notice a big speed improvement from the simple search_solver.
   Contrast the following two tests, which run the two satisfiability solvers
   (search_solver and dpll) on the same CNF sentence:

       python -m unittest test.TestSearchSolverSpeed
       python -m unittest test.TestDpllSpeed

6. All that remains at this point is to augment the SudokuBoard class 
   (in sudoku.py) so that it can use DPLL to solve itself!
   
   Create a method ```.solve()``` of sudoku.SudokuBoard that returns a
   new Sudokuboard instance corresponding to a valid completion of the
   puzzle. For instance, if
   
       board = SudokuBoard([[4, 1, 2, 3], 
                            [3, 4, 1, 2], 
                            [2, 3, 4, 1], 
                            [0, 0, 0, 0]])

   Then ```board.solve()``` should return a SudokuBoard instance equivalent
   to:
   
       SudokuBoard([[4, 1, 2, 3], 
                    [3, 4, 1, 2], 
                    [2, 3, 4, 1], 
                    [1, 2, 3, 4]])
                    
   If there are no valid completions,  then ```.solve()``` should
   return ```None```. If there are multiple valid completions, then any
   may be returned.
   
   Once you have a successful implementation, the following unit tests should
   succeed:
   
       python -m unittest test.TestSudokuBoardSolve

## Conclusion

At this point, you should be able to run ```python youdoku.py``` in a shell and 
use Youdoku to create 2x2 Sudoku puzzles.

If your implementations are not that fast, then it may run quite slowly
(or not at all). My posted solution will be enough to run with only
a little bit of lag on a decent laptop.