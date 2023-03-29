************************************************************************
file with basedata            : c1533_.bas
initial value random generator: 5870
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  132
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       25        6       25
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           6  11
   3        3          3           5   8  12
   4        3          1          15
   5        3          3           7   9  13
   6        3          1          16
   7        3          2          14  16
   8        3          1          15
   9        3          1          10
  10        3          1          17
  11        3          1          13
  12        3          2          14  16
  13        3          2          15  17
  14        3          1          17
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     5       0    9    5    8
         2     7      10    0    4    8
         3    10       0    8    4    7
  3      1     8       4    0    7   10
         2    10       2    0    4    9
         3    10       0    2    5    8
  4      1     2       8    0   10    8
         2     4       0    8    6    8
         3     7       7    0    2    7
  5      1     4       6    0    6    5
         2     4       4    0    6    6
         3     7       3    0    6    2
  6      1     1       0    6    5    7
         2     7       5    0    5    7
         3    10       2    0    5    7
  7      1     6       0    2   10    3
         2    10       4    0    7    3
         3    10       0    2    6    2
  8      1     2      10    0    9    9
         2     6       0    7    8    8
         3     7       8    0    7    6
  9      1     2       0   10    7    8
         2     6       4    0    7    8
         3     8       0   10    5    6
 10      1     2       8    0    7    6
         2     6       6    0    4    6
         3     6       7    0    6    5
 11      1     2       0    7    5    4
         2     5       4    0    5    3
         3     7       4    0    3    3
 12      1     3      10    0    6    7
         2     8       0    8    6    7
         3     9       4    0    4    6
 13      1     3       0    6    6    8
         2     8       7    0    5    7
         3     8       0    4    5    7
 14      1     3       0    5    3   10
         2     4       0    1    3    7
         3     9       8    0    3    7
 15      1     4       8    0    6    5
         2     6       6    0    3    3
         3     6       0    3    3    4
 16      1     6       0   10    4    6
         2     7       7    0    4    6
         3     8       0    8    3    6
 17      1     4       0    3    9    6
         2     7       5    0    9    3
         3    10       3    0    8    2
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   13    8   80   91
************************************************************************
