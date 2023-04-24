************************************************************************
file with basedata            : c1524_.bas
initial value random generator: 819319938
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  113
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       18       11       18
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1          10
   3        3          2           7  11
   4        3          3           5   8   9
   5        3          2           6  15
   6        3          1          10
   7        3          3           9  12  17
   8        3          1          10
   9        3          1          15
  10        3          2          14  16
  11        3          1          12
  12        3          1          13
  13        3          2          15  16
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
  2      1     2      10    9    5    0
         2     6      10    8    3    0
         3     8       9    6    0   10
  3      1     5       9    8   10    0
         2     8       7    5    9    0
         3     9       2    5    0    8
  4      1     5       2    6    3    0
         2     6       2    4    0    4
         3     8       2    2    3    0
  5      1     2      10    8    8    0
         2     8      10    8    0    9
         3     9      10    6    8    0
  6      1     2       6    7    6    0
         2     3       6    6    5    0
         3     4       4    6    0    7
  7      1     3       8    6    0    4
         2     4       8    5    0    3
         3     7       7    2    8    0
  8      1     5       9    2    0    2
         2     5       8    2    0    4
         3     7       7    1    4    0
  9      1     3       2    8    8    0
         2     5       1    6    8    0
         3     9       1    4    7    0
 10      1     1       9    2    0    6
         2     5       9    2    5    0
         3     6       9    2    0    4
 11      1     1       6    8    0    2
         2     2       3    8    4    0
         3     3       3    8    0    1
 12      1     1       9    6    0    9
         2     8       8    5    9    0
         3     9       7    5    0    9
 13      1     1       7   10    0    4
         2     3       5    9    0    3
         3     4       4    8    0    3
 14      1     1       7    2    0    6
         2     1       6    3    0    5
         3     2       5    2    7    0
 15      1     1       6   10    8    0
         2     6       3   10    6    0
         3     9       3   10    2    0
 16      1     1      10    8    0    2
         2     6       5    7    0    2
         3    10       4    6    8    0
 17      1     6       9    4    7    0
         2     8       7    4    0    4
         3     9       7    4    0    3
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   33   24   77   60
************************************************************************