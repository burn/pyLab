<img src="https://img.shields.io/badge/tests-passing-green"> <img
src="https://img.shields.io/badge/lua-yellow"> <img
src="https://img.shields.io/badge/purpose-se--scripting-blueviolet"> <img
src="https://img.shields.io/badge/platform-osx,linux-pink">

<img align=right width=300 src="/img/logo.jpeg">

#  pyLAB = Lean adaptive balancer (Python version)


Given 
- a total evaluation budget N
-  an initial budget N0
-  a pruning factor of P, 

this would explore (say) 10,000 unlabelled items as follows: 

1. use N0 evalaution to label four items ; 
2. divide labelled items into `best` and `rest` using a multi-goal criteria;  
3. build a classifier  to find, for unlabelled item, B=like(`best`), R=like(`resti`); 
4. sort, in ascending order, the remaining    9996 items  by -B/R; 
5. label the  top item (so N=N-1), and
6. discard the last P%   unlabelled items;
7. update best and rest with this new item; 
8. if N &gt; 0, loop back  to (2)
   -  Else terminate, returning the top item in  `best`.

