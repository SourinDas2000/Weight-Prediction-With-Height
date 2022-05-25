 
# The program reads a CSF file containing weight and height data of a 2022 global study. The program aims to study the data and plot a regression line in order to be able to predict weight of an individual based on sex with only height provided. The program uses basic statistics and could succesfully predict weight of a man and a women respectively. The result is an approximate value and a best guess according to the understanding of the data.
  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# Reading the CSV file: 

df = pd.read_csv('Weight height of a sample population.csv')


# Assigning information from the CSV file to a variable:
    
gender = df['Gender']


# Converting the units of the height & weight columns respectively:
    
#### Converting Height Unit From Inches To Centimeter (1 inch = 2,54 cm):
df['Height'] = df['Height'].apply(lambda x: x * 2.54)

#### Converting Weight Unit From Pound To Kilogram (1 lbs = 0.45359237 kg):    
df['Weight'] = df['Weight'].apply(lambda x: x * 0.45359237)


# Seperating male & female data & finding the average:
    
male_data = df[gender == 'Male']
female_data = df[gender == 'Female']

#### Male & Female Height:   
male_height = np.array( male_data['Height'], 
                        dtype= 'float64'
                        )
female_height = np.array( female_data['Height'], 
                          dtype= 'float64'
                          )
                          
#### Male & Female Weight:   
male_weight = np.array( male_data['Weight'], 
                        dtype= 'float64'
                        )
female_weight = np.array( female_data['Weight'], 
                          dtype= 'float64'
                          )
                          
#### Average Height & Weight Of Both:
average_male_height = np.average(male_height)
average_male_weight = np.average(male_weight)
average_female_height = np.average(female_height)
average_female_weight = np.average(female_weight)


# Creating an object for linear regression in order to find the best fit line:
    
#### For Male:        
male_height_weight = linear_model.LinearRegression()
male_height_weight.fit(male_height.reshape(-1,1), male_weight)

#### For Female:    
female_height_weight = linear_model.LinearRegression()
female_height_weight.fit(female_height.reshape(-1,1), female_weight)

#### Finding The Regression Line For Both:    
male_regression_line = male_height_weight.predict(male_height.reshape(-1,1))
female_regression_line = female_height_weight.predict(female_height.reshape(-1,1))


# Prediction of weight with given height:
    
#### For Male:
given_male_height = 180 #cm
predicted_male_weight = male_height_weight.predict(np.array([[given_male_height]]))

#### For Female:
given_female_height = 150 #cm
predicted_female_weight = female_height_weight.predict(np.array([[given_female_height]]))


# Using a matplotlib custom style -- Seaborn:
    
plt.style.use('seaborn')


# Creating two axes:

fig, (ax1,ax2) = plt.subplots( nrows=1, 
                               ncols= 2, 
                               constrained_layout= True
                               )


# Creating a class to plot the data:

class plot(object):
    def __init__(self, axes, x_axis, y_axis):           
     self.axes = axes
     self.x_axis = x_axis    
     self.y_axis = y_axis

#### Function To Plot The Scatter PLot:                  
    def plot_scatter(self):   
        labels = [ '',
                   '', 
                   f'Predicted Weight Is {np.round(predicted_male_weight)[0]} kg for {given_male_height} cm',
                   f'Predicted Weight Is {np.round(predicted_female_weight)[0]} kg for {given_female_height} cm'
                   ]   
        colors = [ 'Red', 
                   'Purple', 
                   'Orange', 
                   'Pink'
                   ]
        for ax, x_axis, y_axis, label, color in zip(self.axes,self.x_axis,self.y_axis,labels,colors):           
            ax.scatter( x_axis, y_axis, 
                         edgecolor= 'black',
                         label= label,
                         color= color, 
                         linewidth= 1, 
                         alpha= 0.75, 
                         s= 75
                         ) 
            self.labelling()
                         
#### Function To Plot The Average Height & Weight:                                        
    def plot_average(self):        
# >>>  Plotting A Vertical & Horizontal Line Denoting The Average In The Plot:
        ymins = [ 50,30
                  ]
        xmins = [ 150,140
                  ]
        labels = [ f'Average Height & Weight: {round(average_male_height)} cm, {round(average_male_weight)} kg',
                  f'Average Height & Weight: {round(average_female_height)} cm, {round(average_female_weight)} kg'
                  ]
        for ax, x_axis, y_axis, xmin, ymin, label in zip(self.axes,self.x_axis,self.y_axis,xmins,ymins,labels):
            ax.vlines( x= x_axis,
                              ymin= ymin,
                              label= label,
                              ymax= y_axis,
                              colors= 'black'
                              )    
            ax.hlines( y= y_axis,
                              xmin= xmin,
                              xmax= x_axis,
                              colors= 'black'
                              )
            self.labelling()
            
#### Function For The Regression Line:
    def plot_regression_line(self):
         colors = [ 'Orange', 
                    'Pink'
                    ]
         for ax, x_axis, y_axis, color in zip(self.axes,self.x_axis,self.y_axis,colors): 
             ax.plot( x_axis, y_axis,
                       color= color,
                       label= 'Line Of Best Fit'
                       )
             self.labelling()

#### Labelling:                                              
    def labelling(self): 
         axis = [ ax1,ax2
                  ]
         titles = [ 'Height And Weight Of Men',
                    'Height And Weight Of Women'
                    ]
         for ax, title in zip(axis, titles): 
             ax.set_title( title,
                           fontsize= 6, 
                           fontweight= 'bold'
                           )          
             ax.set_xlabel( 'Height (cm) >', 
                                    fontsize= 5, 
                                    fontweight= 'bold'
                                    )
             ax.tick_params( axis= 'both', 
                                    labelsize= 5
                                    )
             ax.legend( loc= 'upper left',
                        prop= { 'size': 5, 
                                'weight': 'bold'
                                 }
                        )    
             axis[0].set_ylabel( 'Weight (kg) >', 
                                  fontsize= 5, 
                                  fontweight= 'bold'
                                  )                   
             fig.suptitle( 'Weight Prediction With Height:',
                            fontsize= 8,
                            fontweight= 'bold',
                            fontfamily= 'serif'
                            )
               

# Plotting the data:  

# ...The Code Goes Like: plot(axis, x_axis, y_axis).function()... #

#### Plotting Average: 
plot( [ ax1,ax2
        ], 
      [ average_male_height,
        average_female_height
        ], 
      [ average_male_weight, 
        average_female_weight
        ]
    ).plot_average() 

#### Plotting The Scattered Plot Along With The Predicted Weight:                                 
plot( [ ax1,ax2,ax1,ax2
        ], 
      [ male_height, 
        female_height, 
        given_male_height, 
        given_female_height
        ], 
      [ male_weight, 
        female_weight, 
        predicted_male_weight, 
        predicted_female_weight
        ]
    ).plot_scatter()
    
#### Plotting The Regresssion Line:
plot( [ ax1,ax2
        ], 
      [ male_height, 
        female_height
        ], 
      [ male_regression_line, 
      female_regression_line
      ]
    ).plot_regression_line()


# Showing the plot:
    
plt.show()


''' Created By Sourin Das '''