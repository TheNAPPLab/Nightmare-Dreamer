# def solution(w,h,s):
#     cache = set()
#     arr = [0,1,0,0]
#     rotate_state(arr, cache)
#     print(cache)
    
# def rotate_state(arr, cache):
#     n = len(arr)

#     for k in range(n//2):
#         rotated_arr = arr[k:] + arr[:k]
#         cache.add(''.join(map(str, rotated_arr)))

def solution(w, h, s):
    # Your code here
   
    '''
    Strategy 1 dynamic programming and memoization
    10 ^8 ms
    size of arr 12 * 12 max time to search transpose is 12*12 = 144
    20 possible max states 
    loop through empty arr , either increase state at i,j or move to next index 
    
    (2,3,4)
    0 to 3
    0000 == automatic same with all 1s, all2 all 3 so ans += s
    0000
    cache = (1000000
    
    1000 put in cache after 
    0000
    three choices one move index, move to next state or shift index i +1
    
  first add all the 1s
  1000   1100  1110     1111
  0000   0000  0000 ... 1111 
  
  now back track remove 1 and increase state all the way to max
  1111   1111                                     1111
  1112.. 1113 ... reset now back track two states 1120 
        
    '''
    
    cache = set()
    def dp(i,j,g,state):
        if state not in cache:
            ans +=1
            cache.add(tuple(state))
        state[i][j] = g
        if i + 1 < w:
            for v in g:
                dp(i+1, j, v, state)
        else:
            for v in g:
                dp(0, j+1, v, state)
