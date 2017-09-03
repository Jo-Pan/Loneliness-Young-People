#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 13:27:41 2017

@author: Pan
"""

def areyoulonely():
    print(' ====== LONELINESS TEST  =======') #tab enter to begin
    print('Do you think you are lonely?')
    temp=input() #ignore any input
    print('')
    print('Doesn\'t matter what you think.')
    print('We will see after 8 questions.')
    
    #store user input into each variable:
    x1_public_speaking = int(input('1. Public speaking: Not afraid at all 1-2-3-4-5 Very afraid of (integer)'))
    x2_writing=int(input('2. Poetry writing: Not interested 1-2-3-4-5 Very interested (integer)'))
    x3_internet=int(input('3. Internet: Not interested 1-2-3-4-5 Very interested (integer)'))
    x4_PC=int(input('4. PC Software, Hardware: Not interested 1-2-3-4-5 Very interested (integer)'))
    
    x5_fun_fri=int(input('5. Socializing: Not interested 1-2-3-4-5 Very interested (integer)'))
    x6_eco_man=int(input('6. Economy, Management: Not interested 1-2-3-4-5 Very interested (integer)'))
    x7_cars=int(input('7. Cars: Not interested 1-2-3-4-5 Very interested (integer)'))
    x8_ent_sp=int(input('8. I spend a lot of money on partying and socializing.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)'))    
    #call calulate loneliness function:
    calculateloneliness(x1_public_speaking,x2_writing,x3_internet,x4_PC,x5_fun_fri,x6_eco_man,x7_cars,x8_ent_sp)
    return x1_public_speaking,x2_writing,x3_internet,x4_PC,x5_fun_fri,x6_eco_man,x7_cars,x8_ent_sp

def calculateloneliness(x1,x2,x3,x4,x5,x6,x7,x8):
    #set up the equation:
    yscore=0.00055438+0.133*x1+0.108*x2+0.0924*x3+0.0919*x4-0.102*x5-0.102*x6-0.11*x7-0.128*x8
    print()
    print(' ==========  RESULT  =========')
    if yscore>0.5: #if probability>0.5, then user is lonely
        print('Hi Mr./Ms. Lonely !')
        #print the probability in percent with 2 decimal places
        print('You are '+str('{percent:.2%}'.format(percent=yscore)+' likely to be a lonely person.') )
    else:# else, not lonely
        print('You must be a joyful person!')
        #print the probability in percent with 2 decimal places
        print('You are only'+str('{percent:.2%}'.format(percent=yscore)+' likely to be a lonely person.'))
    
    outputfortest=int(yscore)
    return outputfortest
#call the areyoulonely() function to start test
areyoulonely()

 