from PharmaPy.Commons import trapezoidal_rule
import numpy as np
import json


class Classifier():

    def __init__(self,grid:np.array,distrib:np.array,density:float,trayFilepath:str='',classFilepath:str='',n=10) -> None:

        self.grid = grid
        self.distrib = distrib
        self.density = density
        self.trayFilepath = trayFilepath
        self.classFilepath = classFilepath
        self.n = n
        self._alltrays = None
        self._allclasses = None
        self._alltolerances = None
        
    @property
    def alltrays(self):

        if self._alltrays is None:
            with open(self.trayFilepath) as jsonfile:
                self._alltrays = json.load(jsonfile)['Trays']
        return self._alltrays
    
    @property
    def allclasses(self):
        
        if self._allclasses is None:
            with open(self.classFilepath) as jsonfile:
                contents = json.load(jsonfile)
                self._allclasses = {key: list(value.keys()) for key,value in contents['Classes'].items()}
        return self._allclasses
    
    @property
    def alltolerances(self):

        if self._alltolerances is None:
            with open(self.classFilepath) as jsonfile:
                self._alltolerances = json.load(jsonfile)['Classes']
        return self._alltolerances
    
    def mass_integrator(self,grid:np.array,distrib:np.array,density:float):

        total_weight = 0
        for i in range(1,len(grid)):
            #lin interp
            x = grid[i]
            y = lambda x1: distrib[i-1]+(x1-grid[i-1])*((distrib[i]-distrib[i-1])/(x-grid[i-1]))
            diam = x/2+grid[i-1]/2 #average of xgrid point
            vol = 4/3*np.pi*(diam/2)**3 # assume sphere micron**3
            xs = np.linspace(grid[i-1],x,self.n)
            ys = y(xs)
            counts = trapezoidal_rule(xs,ys)
            total_vol = vol*counts
            total_weight += total_vol*density
        return total_weight
    
    def sieve1(self,size:int,give_remainder=True,give_total=False):

        ## Step 1: Calc total
        if self.grid[0] != 0:
            grid = np.append(np.array([0]),self.grid)
            distrib = np.append(np.array([0]),self.distrib)
        # grid = np.append(np.array([0]),grid)
        # distrib = np.append(np.array([0]),distrib)
        total = self.mass_integrator(grid,distrib,self.density)
        ## step 2 find point where particle <=size
        # This assumes particle same dim as size can pass 
        where_at = np.where(grid<=size)[0] # indices that match description
        ## step 3 find weight fraction
        if len(where_at)>1:
            new_grid = grid[where_at]
            new_distrib = distrib[where_at]
            amount = self.mass_integrator(new_grid,new_distrib,self.density)
            weight_fraction =amount/total
        else:
            weight_fraction = 0
        ## step 4 get remaining distrib and grid
        if not give_remainder:
            return weight_fraction
        try:
            remainder = where_at[-1]
        except IndexError:
            remainder = 0
        if remainder <= len(distrib):
            remainder_distrib = distrib[remainder:]
            remainder_grid = grid[remainder:]
        else:
            remainder_distrib = None
            remainder_grid = None
        if not give_total:
            return weight_fraction,remainder_grid,remainder_distrib
        return weight_fraction,remainder_grid,remainder_distrib, total
    
    def sieve2(self,old_total:float,old_frac:float,size:int,grid:np.array,distrib:np.array,\
               give_remainder=True,give_new_total=False):
        
        ## step 2 find point where particle <=size
        # This assumes particle same dim as size can pass 
        where_at = np.where(grid<=size)[0] # indices that match description
        ## step 3 find weight fraction
        if len(where_at)>1:
            new_grid = grid[where_at]
            new_distrib = distrib[where_at]
            amount = self.mass_integrator(new_grid,new_distrib,self.density)
            weight_fraction =amount/old_total
        else:
            weight_fraction = 0
        ## step 4 get remaining distrib and grid
        weight_fraction += old_frac
        if not give_remainder:
            return weight_fraction
        try:
            remainder = where_at[-1]
        except IndexError:
            remainder = 0
        if remainder <= len(distrib):
            remainder_distrib = distrib[remainder:]
            remainder_grid = grid[remainder:]
        else:
            remainder_distrib = None
            remainder_grid = None
        if not give_new_total:
            return weight_fraction,remainder_grid,remainder_distrib
        return weight_fraction,remainder_grid,remainder_distrib, self.mass_integrator(grid,distrib,self.density)
    
    def get_trays(self,chosentrays:list) -> tuple :
        
        trays = {}
        for key in chosentrays:
            try:
                trays[key] = self.alltrays[key]
            except KeyError:
                print(f'Chosen tray size does not exist: {key}')
        #ensure that the trays are in the right order
        sizes = list(trays.values())
        sizes.sort()
        
        # create inverted dictionary to pair fractions with corect sieve
        trays_inv = {v: k for k, v in trays.items()}
        return sizes,trays,trays_inv
    
    def get_classes(self,chosen_classes:list) -> tuple:

        classes = {}
        for key in chosen_classes:
            if key.lower() == 'all':
                classes[key] = list(self.alltrays.keys())
                continue
            try:
                classes[key] = self.allclasses[key] #list of tray sizes
            except KeyError:
                print(f'Chosen class does not exist: {key}')
        classes_inv = {tuple(v): k for k, v in classes.items()}
        return classes, classes_inv
    
    def run(self,chosen_classes:list)->dict:

        '''Breaks down the CSD into the segmentations specified for each class.
        chosen_classes: list of the desired classes
        '''
        chosen_classes= [str(v) for v in chosen_classes]
        results = {str(cl):None for cl in chosen_classes}
        classes,classes_inv = self.get_classes(chosen_classes) 
        
        for i,cl in enumerate(chosen_classes):
            sizes,trays,trays_inv = self.get_trays(classes[str(cl)])
            result = {tray:0 for tray in trays.keys()}
            passing_fraction,new_grid,new_distrib,total = self.sieve1(sizes[0],give_total=True) #sieve1 is needed because there is a hidden integral bound of 0->first bin size
            result[trays_inv[sizes[0]]] = passing_fraction
            for ii,size in enumerate(sizes):
                if ii==0:continue
                passing_fraction,new_grid,new_distrib = self.sieve2(total,passing_fraction,size,new_grid,new_distrib) #sieve2 does not start at 0 and caluclates weight based on the leftover distribution
                result[trays_inv[size]] = passing_fraction
            
            results[str(cl)] = result
        return results
    
    def diagnose(self,verbose = 1):
        results = self.run(['all'])['all']
        if verbose>1:
            print(results)
        result2 = {key:val for key,val in results.items() if val <.99}
        if verbose ==1:
            print(result2)
        return result2
    
    def inbetween(self, value:float,listed_tol:list,tol:float=1e-4)->bool:

        '''Checks if value is inbetween the listed tolerances, or close enough (according to tol)'''
        listed_tol = np.array(listed_tol)/100
        if len(listed_tol)==1:
            lower = listed_tol[0]-tol
            upper = listed_tol[0]+tol
        else:
            lower = listed_tol[0]-tol
            upper = listed_tol[1]+tol
        return lower <= value <= upper
    
    def score(self, values:dict, listed_tol:dict, verbose = False):

        '''Gives the total percentage error between a CSD and the target class.

        values: dict of the tray:value created by self.run
        listed_tol: dict of the tray:list(tols) for that class
        verbose: bool True if the percentage breakdown for each tray is also desired
        
        returns:
        summation of the percent distance from fulfilling the tolerances for each tray. A value of 0 means the 
        CSD is described by the target class size.'''
        errors = []
        breakdown = {}
        for tray, tol in listed_tol.items():
            if self.inbetween(values[tray],tol):
                error = 0
            else:
                index = 0 if values[tray]*100 < tol[0] else 1
                error = abs(values[tray]*100-tol[index])/tol[index]
            errors.append(error)
            breakdown[tray] = error
        sumPercError = sum(errors)
        if verbose:
            return sumPercError, breakdown
        return sumPercError
            
    def classify(self,chosen_classes:list): 

        '''Auto matches the classes that the distribution complies with. If it does not fit any class, 
        this is stated and the breakdown from self.run() is displayed'''
        chosen_classes= [str(v) for v in chosen_classes]
        results = self.run(chosen_classes)
        matches = {str(cl):[] for cl in chosen_classes}
        for cl in results.keys():
            tols = self.alltolerances[str(cl)]
            matches[str(cl)] = [self.inbetween(results[cl][tray],tols[tray]) for tray in results[cl]]
        one = False
        for cl in matches.keys():
            if all(matches[cl]):
                print(cl, 'matched')
                one = True
        if not one:
            print('No matching classes for given distribution')
            print('Results:\n')
            print(results)
            print(matches)
    
    def quantify(self, target_class):
        
        '''Quantifies how close the CSD is to being the target class as measured by the summed percentage error(SPE).

        SPE is the summation of the percent distance from fulfilling the tolerances for each tray. A value of 0 means the 
        CSD is described by the target class size.'''
        results = self.run([target_class])
        tols = self.alltolerances[str(target_class)]
        sumPercError = self.score(results[str(target_class)],tols)
        return sumPercError*100


if __name__ == "__main__":
    import pickle
    with open('c:\\Users\\zhillma\\OneDrive - purdue.edu\\Documents\\Documents\\_Grad_School\\mypharmadev\\Scripts\\cryst_data.p','rb') as file:
        crystsdistrib = pickle.load(file)
    test_grid = np.arange(1,501)
    test_distrib = crystsdistrib
    density = 1
    tray_file = r'C:\Users\zhillma\OneDrive - purdue.edu\Documents\Documents\_Grad_School\mypharmadev\Data\Sieve_trays.json'
    calss_file = r'C:\Users\zhillma\OneDrive - purdue.edu\Documents\Documents\_Grad_School\mypharmadev\Data\class_sizes2.json'

    ins = Classifier(test_grid,test_distrib,density,tray_file,calss_file)#,tray_file,calss_file)
    fraction,new_grid,new_distrib,total =ins.sieve1(44,give_total=True)
    val = ins.quantify(1)
    print("done", val)