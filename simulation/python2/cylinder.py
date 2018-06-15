'''Provides a class for representing data returned from the cylinder fitter.'''

class Cylinder():


  def __init__(self, center, axis, radius, height):
    '''Creates a Grasp object with everything needed.'''

    self.center = center
    self.axis = axis
    self.radius = radius
    self.height = height