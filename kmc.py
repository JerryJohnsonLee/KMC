'''
Author: Jie Li
Created: April 21, 2019
'''

import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import pickle
import time

colors=("red","blue","green","purple","orange")
markers=("o","^","s","*","h","D")
configs=[(c,m) for c in colors for m in markers]
random.shuffle(configs)
snapshop_saving_file="snapshots.pkl"
real_time=time.time()

class Lattice():
    def __init__(self,x_length,y_length,lattice="r"):
        self.X=x_length
        self.Y=y_length
        if lattice=="r":
            # Rectangular lattice
            self.hex=0
        elif lattice=="h":
            # Hexagonal lattice
            self.hex=1
        self.grids=np.zeros((self.X,self.Y),dtype=int)
        self.species={0:""}
        self.species_count=0
        self.species_query={"":0}

    def neighbor(self,x_pos,y_pos):
        '''Return the periodic coordinates and species identifier for all the neighbor positions'''
        per_xm=(x_pos-1)%self.X # Periodic x minus 1
        per_xp=(x_pos+1)%self.X
        per_ym=(y_pos-1)%self.Y
        per_yp=(y_pos+1)%self.Y
        if self.hex:
            return [((per_xm,y_pos),self.grids[per_xm,y_pos]),((per_xp,y_pos),self.grids[per_xp,y_pos]),
            ((x_pos,per_ym),self.grids[x_pos,per_ym]),((x_pos,per_yp),self.grids[x_pos,per_yp]),
            ((per_xp,per_ym),self.grids[per_xp,per_ym]),((per_xm,per_yp),self.grids[per_xm,per_yp])]
        else:
            return [((per_xm,y_pos),self.grids[per_xm,y_pos]),((per_xp,y_pos),self.grids[per_xp,y_pos]),
            ((x_pos,per_ym),self.grids[x_pos,per_ym]),((x_pos,per_yp),self.grids[x_pos,per_yp])]

    def add_species(self,species):
        if species=="*":
            species="" # Change the star representation to null representation
        if not species in self.species_query:
            self.species_count+=1
            self.species[self.species_count]=species
            self.species_query[species]=self.species_count

    def initialize_grid(self,species_id,theta,mode="deterministic"):
        '''Deterministic mode: the coverage is exactly equal to theta
        Random mode: Each grid point has a chance of being occupied equal to theta'''
        if mode=="random":
            for i in range(self.X):
                for j in range(self.Y):
                    if np.random.random()<theta:
                        self.grids[i,j]=species_id
        elif mode=="deterministic":
            all_pos=[(i,j) for i in range(self.X) for j in range(self.Y)]
            line=int(theta*self.X*self.Y)
            random.shuffle(all_pos)
            for i,j in all_pos[:line]:
                self.grids[i,j]=species_id

    def adsorption(self,species_id):
        # Should change method with coverage
        species=self.get_species_pos()
        if len(species[""])==0:
            return False
        else:
            pos=random.choice(species[""])
            self.grids[pos]=species_id
            return True

    def desorption(self,species_id):
        # Should change method with coverage
        species=self.get_species_pos()
        if len(species[self.species[species_id]])==0:
            return False
        else:
            pos=random.choice(species[self.species[species_id]])
            self.grids[pos]=0
            return True

    def diffuse(self,species_id):
        species=self.get_species_pos()
        if len(species[self.species[species_id]])==0:
            return False
        else:
            pos=random.choice(species[self.species[species_id]])
            neighbors=self.neighbor(*pos)
            if not 0 in [nbr[1] for nbr in neighbors]:
                return False
            else:
                choices=[nbr[0] for nbr in neighbors if nbr[1]==0]
                choice=random.choice(choices)
                self.grids[pos]=0
                self.grids[choice]=species_id
                return True


    def get_species_pos(self,grid_array=None):
        if grid_array is None:
            grid_array=self.grids
        species_dict={species:[] for species in self.species.values()}
        for i in range(self.X):
            for j in range(self.Y):
                species_dict[self.species[grid_array[i,j]]].append((i,j))
        return species_dict

    def set_species(self,species,pos):
        if species=="*":
            species=""
        self.grids[pos]=self.species_query[species]

    def set_config(self,configuration):
        if configuration.shape[0]==self.X and configuration.shape[1]==self.Y:
            self.grids=configuration

    def show_grid(self,grid_array=None,title=None):
        if len(configs)<self.species_count:
            print("Too many species!")
        else:
            config=configs[:self.species_count] # Vacancy will not be shown
            if grid_array is None:
                grid_array=self.grids
            species=self.get_species_pos(grid_array)
            del species[""]
            if self.hex:
                fig=plt.figure()
                # Draw the species
                for spec,conf in zip(species,config):
                    if len(species[spec])==0:
                        continue
                    coords=np.array(species[spec])
                    # In hexagonal lattice, (i',j')=(i+j/2,√3/2*j)
                    plt.scatter(coords[:,0]+coords[:,1]/2,coords[:,1]*np.sqrt(3)/2,c=conf[0],marker=conf[1],label=spec,s=100)
                for i in range(self.X):
                    # / shape lines
                    plt.plot([i,i+(self.Y-1)/2],[0,(self.Y-1)*np.sqrt(3)/2],c="k",linewidth=0.5)
                for j in range(self.Y):
                    # Horizontal lines
                    plt.plot([j/2,j/2+self.X-1],[j*np.sqrt(3)/2,j*np.sqrt(3)/2],c="k",linewidth=0.5)
                # \ shape lines
                for k in range(min(self.X,self.Y)-1):
                    # Lower triangular lines
                    p1i,p1j=min(k+1,self.X),k+1-min(k+1,self.X)
                    p2i,p2j=k+1-min(k+1,self.Y-1),min(k+1,self.Y-1)
                    plt.plot([p1i+p1j/2,p2i+p2j/2],[p1j*np.sqrt(3)/2,p2j*np.sqrt(3)/2],c="k",linewidth=0.5)
                for k in range(min(self.X,self.Y)-1):
                    # Upper triangular lines
                    p1i,p1j=self.X-1,self.Y-min(k+1,self.X)-1
                    p2i,p2j=self.X-1-min(k+1,self.Y-1),self.Y-1
                    plt.plot([p1i+p1j/2,p2i+p2j/2],[p1j*np.sqrt(3)/2,p2j*np.sqrt(3)/2],c="k",linewidth=0.5)
                for k in range(max(self.X,self.Y)-min(self.X,self.Y)-1):
                    # Middle Lines
                    if self.X<self.Y:
                        p1i,p1j=self.X-1,k+1
                        p2i,p2j=0,self.X+k
                    elif self.X>self.Y:
                        p1i,p1j=self.Y+k,0
                        p2i,p2j=k+1,self.Y-1
                    else:
                        p1i,p1j=self.X-1,0
                        p2i,p2j=0,self.Y-1
                    plt.plot([p1i+p1j/2,p2i+p2j/2],[p1j*np.sqrt(3)/2,p2j*np.sqrt(3)/2],c="k",linewidth=0.5)
                plt.legend()
                fig.gca().axis("equal")
                if not title is None:
                    plt.title(title)
                plt.show()
            else:
                fig=plt.figure()
                for i in range(self.X):
                    plt.plot([i,i],[0,self.Y-1],c="k",linewidth=0.5)
                for j in range(self.Y):
                    plt.plot([0,self.X-1],[j,j],c="k",linewidth=0.5)
                for spec,conf in zip(species,config):
                    if len(species[spec])==0:
                        continue
                    coords=np.array(species[spec])
                    plt.scatter(coords[:,0],coords[:,1],c=conf[0],marker=conf[1],label=spec,s=100)
                plt.legend()
                fig.gca().axis("equal")
                if not title is None:
                    plt.title(title)
                plt.show()

class KMC_simulator():
    def __init__(self):
        self.lattice=None
        self.allowed_reactions=[]
        self.reaction_count=0
        self.external={}
        self.temp=0
        self.time=0

    def initialize(self,x_length=None,y_length=None,lattice="r",temperature=None):
        '''lattice: r - rectangular lattice, h - hexagonal lattice'''
        if self.lattice==None:
            self.lattice=Lattice(x_length,y_length,lattice)
        if not temperature==None:
            self.temp=temperature
        self.time=0
        self.snapshots=[]

    def add_reaction(self,reaction_type=None,reaction_formula=None,activation=None,temperature=None,additional_vars=None,reaction=None):
        '''Reaction types: dif - diffusion, ads1 - single site adsorption, ads2 - double site adsorption
        surf2 - surface reaction with two species involved, desr - reactive desorption, 
        des1 - single site desorption, des2 - double site desorption
        activation in units of eV
        temperature in units of K'''
        if reaction==None:
            if temperature==None:
                temperature=self.temp
            reaction=Reaction(reaction_type,activation,temperature)
            reaction.parse_reaction(reaction_formula)
            if reaction_type in {"ads1","ads2"}:
                for substance in self.external:
                    if substance.name==reaction.reactants[0] and substance.fixed:
                        reaction.register_variable("p",substance.amount)
                        break
            if not additional_vars is None:
                for key in additional_vars:
                    reaction.register_variable(key,additional_vars[key])
        self.allowed_reactions.append(reaction)
        for species in reaction.reactants:
            if "*" in species:
                self.lattice.add_species(species)
        for species in reaction.products:
            if "*" in species:
                self.lattice.add_species(species)


    def add_external_substance(self,substance=None,substance_name=None):
        if not substance is None:
            self.external[substance.name]=substance
            for reaction in self.allowed_reactions:
                if reaction.type in {"ads1","ads2"}:
                    if reaction.reactants[0]==substance.name:
                        reaction.register_variable("p",substance.amount)
            return
        if not substance_name is None:
            if substance_name in self.external:
                if not self.external[substance_name].fixed:
                    self.external[substance_name].amount+=1
            else:
                self.external[substance_name]=Species(substance_name,1,False)        


    def find_reactions(self):
        potential_reaction=[]
        potential_reactant=[]
        reaction_rate=[]
        fingerprints=[]
        lattice_species=self.lattice.get_species_pos()
        for reactant in self.external:
            # External reactants can only adsorb onto lattice
            for reaction in self.allowed_reactions:
                if reaction.stoichiometry(reactant):
                    # The reaction involves this specific reactant
                    if reaction.type=="ads1":
                        possible_pos_count=len(lattice_species[""])
                        potential_reaction.extend([reaction]*possible_pos_count)
                        potential_reactant.extend([(pos,) for pos in lattice_species[""]]) # Add all vacancies
                        reaction_rate.extend([reaction.rate()]*possible_pos_count)
                    elif reaction.type=="ads2":
                        pre_length=len(potential_reactant)
                        for vacancy1 in lattice_species[""]:
                            for neighbor_pos,neighbor_subs in self.lattice.neighbor(*vacancy1):
                                if neighbor_subs==self.lattice.species_query[""]:
                                    fingerprint=Reaction.generate_fingerprint(("*","*"),(vacancy1,neighbor_pos))
                                    if not fingerprint in fingerprints: 
                                        potential_reactant.append((vacancy1,neighbor_pos)) # Whether we should consider 2 different adhesive species?
                                        fingerprints.append(fingerprint)
                        possible_pos_count=len(potential_reactant)-pre_length
                        potential_reaction.extend([reaction]*possible_pos_count)
                        reaction_rate.extend([reaction.rate()]*possible_pos_count)
        del lattice_species[""]
        for substance in lattice_species:
            if len(lattice_species[substance])==0:
                continue
            # On lattice reactions
            for reaction in self.allowed_reactions:
                if reaction.stoichiometry(substance):
                    # Prepare dictionary recording all the other atoms (and number) needed for the reaction
                    reactants={reactant:count for reactant,count in zip(reaction.reactants,reaction.reactants_coefs)}
                    if "*" in reactants:
                        reactants[""]=reactants["*"]
                        del reactants["*"]
                    if reactants[substance]==1:
                        del reactants[substance]
                    else:
                        reactants[substance]-=1
                    # The reaction involves this specific reactant
                    for reactant_pos in lattice_species[substance]:
                        # Single reactant position
                        if reaction.type in {"dif","surf2","des2"}:
                            for neighbor_pos,neighbor_subs in self.lattice.neighbor(*reactant_pos):
                                if self.lattice.species[neighbor_subs]==list(reactants)[0]:
                                    # Neighbor allow for reaction
                                    fingerprint=Reaction.generate_fingerprint((substance,list(reactants)[0]),(reactant_pos,neighbor_pos))
                                    if not fingerprint in fingerprints:
                                        # New microscopic reaction discovered
                                        potential_reaction.append(reaction)
                                        potential_reactant.append((reactant_pos,neighbor_pos))
                                        reaction_rate.append(reaction.rate())
                                        fingerprints.append(fingerprint)
                        elif reaction.type=="des1":
                            potential_reaction.append(reaction)
                            potential_reactant.append((reactant_pos,))
                            reaction_rate.append(reaction.rate())
                        else:
                            # Reactive desorption
                            neighbors={}
                            for neighbor_pos,neighbor_subs in self.lattice.neighbor(*reactant_pos):
                                if neighbor_subs in neighbors:
                                    neighbors[neighbor_subs].append(neighbor_pos)
                                else:
                                    neighbors[neighbor_subs]=[neighbor_pos]
                            allowed=True
                            for reactant in reactants:
                                reactant_code=self.lattice.species_query[reactant]
                                if not reactant_code in neighbors or len(neighbors[reactant_code])<reactants[reactant]:
                                    allowed=False
                                    break
                            if allowed:
                                current_record=[(reactant_pos,)]
                                current_substance_record=[(substance,)]
                                for reactant in reactants:
                                    temporal_record=[]
                                    temporal_substance_record=[]
                                    for pos in list(itertools.combinations(neighbors[self.lattice.species_query[reactant]],reactants[reactant])):
                                        for item in current_record:
                                            temporal_record.append(item+pos)
                                        for item in current_substance_record:
                                            temporal_substance_record.append(item+(reactant,)*reactants[reactant])
                                    current_record=temporal_record
                                    current_substance_record=temporal_substance_record
                                for pos,subs in zip(current_record,current_substance_record):
                                    fingerprint=Reaction.generate_fingerprint(subs,pos)
                                    if not fingerprint in fingerprints:
                                        potential_reaction.append(reaction)
                                        potential_reactant.append(pos)
                                        reaction_rate.append(reaction.rate())
                                        fingerprints.append(fingerprint)
        return potential_reaction,potential_reactant,reaction_rate

    @staticmethod
    def choose_random(array):
        cumulative=[np.sum(array[:i+1]) for i in range(len(array))]
        pointer=np.random.random()*cumulative[-1]
        for idx,num in enumerate(cumulative):
            if num>pointer:
                return idx,cumulative[-1]

    def start(self,steps=1,show_steps_interval=500,snapshot_interval=1000):
        for n in range(steps):
            if (n+1)%show_steps_interval==0:
                print("Step ",n+1,",Speed %.3e s/hr"%(self.time*3600/(time.time()-real_time)))
            self.progress()
            if (n+1)%snapshot_interval==0:
                # Save snapshot files
                self.generate_snapshot()
                with open(snapshop_saving_file,"wb") as f:
                    pickle.dump({"lattice":self.lattice,"history":self.snapshots},f)


    def generate_snapshot(self):
        current=Snapshot(self.time,self.temp)
        current.snapshot_lattice(self.lattice)
        for species in self.external:
            current.snapshot_external_substance(self.external[species])
        self.snapshots.append(current)

    def progress(self):
        potential_reaction,potential_reactant,reaction_rate=self.find_reactions()
        chosen_rxn,k_tot=KMC_simulator.choose_random(reaction_rate)
        rxn=potential_reaction[chosen_rxn]
        reactant=list(potential_reactant[chosen_rxn])
        # Conduct the reaction
        if rxn.type=="ads1":
            self.lattice.set_species(rxn.products[0],reactant[0])
        elif rxn.type=="ads2":
            if len(rxn.products)==1:
                # The adhesive species are the same
                for reactant_pos in reactant:
                    self.lattice.set_species(rxn.products[0],reactant_pos)
            else:
                raise NotImplementedError()
        elif rxn.type=="dif":
            temp=self.lattice.grids[reactant[0]]
            self.lattice.grids[reactant[0]]=self.lattice.grids[reactant[1]]
            self.lattice.grids[reactant[1]]=temp
        elif rxn.type=="surf2":
            if len(rxn.products)==1:
                positions=reactant.copy()
                random.shuffle(positions) # select a random place from the original positions for the product
                self.lattice.set_species(rxn.products[0],positions[0])
                self.lattice.set_species("",positions[1])
            else:
                self.lattice.set_species(rxn.products[0],reactant[0])
                self.lattice.set_species(rxn.products[1],reactant[1])
        elif rxn.type in {"des1","des2","desr"}:
            for reactant_pos in reactant:
                self.lattice.set_species("",reactant_pos)
            self.add_external_substance(substance_name=rxn.products[0])
        self.time-=1/k_tot*np.log(np.random.random())


        



class Species():
    def __init__(self,name,amount,fixed_amount=True):
        self.name=name
        self.amount=amount
        self.fixed=fixed_amount

class Snapshot():
    def __init__(self,time,temperature):
        self.time=time
        self.temp=temperature
        self.external={}
        self.grid=None

    def snapshot_external_substance(self,species):
        self.external[species.name]=species.amount

    def snapshot_lattice(self,lattice):
        self.grid=lattice.grids.copy()
        

class Reaction():
    def __init__(self,reaction_type,activation,temperature):        
        self.type=reaction_type
        self.reactants=[]
        self.reactants_coefs=[]
        self.products=[]
        self.product_coefs=[]
        self.activation=activation
        self.temp=temperature
        self.variables={}

    @staticmethod
    def generate_fingerprint(substances,positions):
        sorted_list=sorted(zip(substances,positions))
        return tuple(sorted_list)
    
    def parse_reaction(self,reaction):
        '''Parses a reaction string like: H2+*->2*'''
        all_reactants,all_products=reaction.split("->")
        for reactant in all_reactants.split("+"):
            if reactant[0].isnumeric():
                first_idx=[c.isnumeric() for c in reactant].index(False)
                count=int(reactant[:first_idx])
                substance=reactant[first_idx:]
            else:
                count=1
                substance=reactant
            self.reactants.append(substance)
            self.reactants_coefs.append(count)
        for product in all_products.split("+"):
            if product[0].isnumeric():
                first_idx=[c.isnumeric() for c in product].index(False)
                count=int(product[:first_idx])
                substance=product[first_idx:]
            else:
                count=1
                substance=product
            self.products.append(substance)
            self.product_coefs.append(count)

    def register_variable(self,label,value):
        self.variables[label]=value

    def rate(self):
        # k_0*k_B T/h*exp(-E_a/k_B T) approx to 5e12*exp(-E_a/k_B T)
        if self.type in {"ads1","ads2"}:
            if not "A" in self.variables:
                while True:
                    accepted_input=input("Calculation of reaction rate for "+self.get_reaction_expression()+
                    " needs reaction site area A:")
                    if accepted_input.replace(".","").isnumeric():
                        self.variables["A"]=float(accepted_input)
                        break
            if not "m" in self.variables:
                while True:
                    accepted_input=input("Calculation of reaction rate for "+self.get_reaction_expression()+
                    " needs the reactant mass m:")
                    if accepted_input.replace(".","").isnumeric():
                        self.variables["m"]=float(accepted_input)
                        break
            return self.variables["p"]*self.variables["A"]/np.sqrt(2*np.pi*self.variables["m"]*1.3806e-23*self.temp)
        elif self.type in {"des1","des2"}:
            # Here the parathesis after self.variables["k_ads"] allows rate to update dynamically
            return self.variables["k_ads"]()*np.exp(-self.activation/(8.6195e-5*self.temp))
        else:
            return 2.0837e10*self.temp*np.exp(-self.activation/(8.6195e-5*self.temp))

    def stoichiometry(self,reactant):
        if reactant in self.reactants:
            return self.reactants_coefs[self.reactants.index(reactant)]
        else:
            return 0

    def get_reaction_expression(self):
        reactants_list=[(str(coef) if coef>1 else "")+species for coef,species in zip(self.reactants_coefs,self.reactants)]
        products_list=[(str(coef) if coef>1 else "")+species for coef,species in zip(self.product_coefs,self.products)]
        return "+".join(reactants_list)+"->"+ "+".join(products_list)


if __name__=="__main__":
    simulator=KMC_simulator()
    simulator.initialize(10,10,"h",600)

    ads1=Reaction("ads1",-0.5,simulator.temp)
    ads1.parse_reaction("A+*->A*")
    ads1.register_variable("A",2.5981e-20)
    ads1.register_variable("m",4.6833e-26)
    simulator.add_reaction(reaction=ads1)

    des1=Reaction("des1",0.5,simulator.temp)
    des1.parse_reaction("A*->A+*")
    des1.register_variable("k_ads",ads1.rate)
    simulator.add_reaction(reaction=des1)

    ads2=Reaction("ads2",-0.6,simulator.temp)
    ads2.parse_reaction("H2+2*->2H*")
    ads2.register_variable("A",5.1962e-20)
    ads2.register_variable("m",3.3452e-27)
    simulator.add_reaction(reaction=ads2)

    des2=Reaction("des2",0.6,simulator.temp)
    des2.parse_reaction("2H*->H2+2*")
    des2.register_variable("k_ads",ads2.rate)
    simulator.add_reaction(reaction=des2)

    simulator.add_reaction("surf2","A*+H*->AH*+*",1.24)
    simulator.add_reaction("surf2","AH*+*->A*+H*",0.94)
    simulator.add_reaction("desr","AH*+H*->AH2+2*",1.18)
    simulator.add_reaction("dif","A*+*->*+A*",0.4)
    simulator.add_reaction("dif","H*+*->*+H*",0.3)
    simulator.add_reaction("dif","AH*+*->*+AH*",0.45)
    simulator.add_external_substance(Species("A",1e6))
    simulator.add_external_substance(Species("H2",1e6))
    simulator.start(1000000000,1000)

