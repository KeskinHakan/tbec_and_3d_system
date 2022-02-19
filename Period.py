# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 17:02:57 2022

@author: hakan
"""

##################################################################
## 3D frame example to show how to render opensees model and 
## plot mode shapes
##
## By - Hakan Keskin, PhD Student, Istanbul Technical University.
## Updated - 23/10/2021
##################################################################

import openseespy.postprocessing.Get_Rendering as opsplt
import openseespy.opensees as ops
import streamlit as st
from pandas import read_excel
import pandas as pd
import pydeck as pdk
from math import asin, sqrt
import openseespy.postprocessing.ops_vis as opsv
import matplotlib.pyplot as plt
from Deneme_Spectrum_Streamlit import soilType_func
from numpy import arange
from geopy.geocoders import Nominatim


# Initialize Nominatim API
geolocator = Nominatim(user_agent="geoapiExercises")

st.title("Turkish Building Seismic Code (TBEC) - Calculation App")

design_type = st.sidebar.selectbox("Design Type: ", {"Nonlinear Design", "Elastic Design"})

def design_choice():
    return design_type

if design_type == "Elastic Design":
    switch = 3
    st.warning("You need to upload a csv or excel file.")
    main_file_name = st.file_uploader("Choose a file")
    
    """
    This application was developed to find out design (SaR) and elastic spectral acceleration (SaE) of the structure
    that located in Turkey. 
    
    To use the app you should choose at the following steps below:
        
        1- Write the exact location of the structure.
        2- Type of structure
        3- Ductility of structure
        4- Structure's supporting system against earthquake loads.
        5- Building importance Factor
        6- Soil Type
        7- Fundamental period of the structure that you want to find exact
            SaR and SaE values.
            
    To find the more information about these parameters please check the TBEC 2018
    Chapter 2, 3 & 4. You can download the TBEC 2018 in Turkish format at the following
    link: https://www.imo.org.tr/resimler/dosya_ekler/89227ad223d3b7a_ek.pdf
        
    After these choices this app will give the design point for fundamental period of the structure.
    
    Note: These results calculated with enterpolation of the spectral values of the coordinates that
    shared by AFAD (https://tdth.afad.gov.tr/TDTH/main.xhtml). These results cannot use for the design or any studies
    please be sure the find design parameters from the AFAD system. This app just prepared to give an information.
    
    """
    
    if main_file_name is not None:
        
        section_rebar = "Frame Props 02 - Concrete Col" # change it to the name of your excel file
        section_properties = 'Frame Props 01 - General'
        restrained = 'Joint Restraint Assignments'
        frame_section = 'Frame Section Assignments'
        assembled_joint_masses = 'Assembled Joint Masses' # change it to your sheet name
        connectivity_frame = "Connectivity - Frame"
        joint_coordinates = "Joint Coordinates"
            
        
        # main_file_name = st.file_uploader("Choose a file")
        df_joint_mass = read_excel(main_file_name, sheet_name = assembled_joint_masses)
        df_joint = read_excel(main_file_name, sheet_name = joint_coordinates)
        df_frame = read_excel(main_file_name, sheet_name = connectivity_frame)
        df_restrain = read_excel(main_file_name, sheet_name = restrained)
        df_frame_section = read_excel(main_file_name, sheet_name = frame_section)
        df_section_properties = read_excel(main_file_name, sheet_name = section_properties)
        df_section_rebar = read_excel(main_file_name, sheet_name = section_rebar)
        

        
        # main_file_name = '3D_Example_Deneme.xlsx' # change it to the name of your excel file
        
        # # df_main = read_excel(main_file_name, sheet_name = my_sheet_main, engine='openpyxl')
        # df_joint_mass = read_excel(main_file_name, sheet_name = assembled_joint_masses)
        # df_joint = read_excel(main_file_name, sheet_name = joint_coordinates)
        # df_frame = read_excel(main_file_name, sheet_name = connectivity_frame)
        # df_restrain = read_excel(main_file_name, sheet_name = restrained)
        # df_frame_section = read_excel(main_file_name, sheet_name = frame_section)
        # df_section_properties = read_excel(main_file_name, sheet_name = section_properties)
        # df_section_rebar = read_excel(main_file_name, sheet_name = section_rebar)
        # # print(df_main.head()) # shows headers with top 5 rows
        # # print(df_main.info())
        
        df_section_rebar.columns = df_section_rebar.iloc[0] 
        df_section_rebar = df_section_rebar[2:]
        df_section_rebar.reset_index(inplace = True, drop = True)
        
        long_dia = df_section_rebar["BarSizeL"]
        len_long_dia = len(long_dia)
        long_diameter = []
        
        stirrup_dia = df_section_rebar["BarSizeC"]
        
        df_frame.columns = df_frame.iloc[0] 
        df_frame = df_frame[2:]
        df_frame.reset_index(inplace = True, drop = True)
        
        df_restrain.columns = df_restrain.iloc[0] 
        df_restrain = df_restrain[2:]
        df_restrain.reset_index(inplace = True, drop = True)
        
        df_frame_section.columns = df_frame_section.iloc[0] 
        df_frame_section = df_frame_section[2:]
        df_frame_section.reset_index(inplace = True, drop = True)
        
        df_section_properties.columns = df_section_properties.iloc[0] 
        df_section_properties = df_section_properties[2:]
        df_section_properties.reset_index(inplace = True, drop = True)
        
        df_joint= df_joint.drop(df_joint.index[[1]])
        df_joint.reset_index(inplace = True, drop = True)
        df_joint.columns = df_joint.iloc[0] 
        df_joint = df_joint[1:]
        
        total_joint_rows = len(df_joint.axes[0])
        total_frame_rows = len(df_frame.axes[0]) 
        total_restrain_rows = len(df_restrain.axes[0]) 
        
        df_joint_mass= df_joint_mass.drop(df_joint_mass.index[[1]])
        df_joint_mass.reset_index(inplace = True, drop = True)
        df_joint_mass.columns = df_joint_mass.iloc[0] 
        df_joint_mass = df_joint_mass[1:]
        
        stirrup_diameter = []
        
        i = 1
        while i <= len_long_dia:
            diameter_long = long_dia.iloc[i-1]
            diameter_long = diameter_long.replace("d", " ")
            long_diameter.append(diameter_long)
            
            diameter_stirrup = stirrup_dia.iloc[i-1]
            diameter_stirrup = diameter_stirrup.replace("d", " ")
            stirrup_diameter.append(diameter_stirrup)
        
            i = i+1
            
        long_diameter = pd.DataFrame(long_diameter)
        long_dia = long_diameter.select_dtypes(include='object').columns
        long_diameter[long_dia] = long_diameter[long_dia].astype("int")  
        
        stirrup_diameter = pd.DataFrame(stirrup_diameter)
        str_dia = stirrup_diameter.select_dtypes(include='object').columns
        stirrup_diameter[str_dia] = stirrup_diameter[str_dia].astype("int")   
        
        df_section_rebar['BarSizeL'] = long_diameter[0].values
        df_section_rebar['BarSizeC'] = stirrup_diameter[0].values
        
        # st.sidebar.header("Geometry of Structure")
        # numBayX = st.sidebar.number_input("Number of Bay - X: ", value=1, step=1)
        # numBayY = st.sidebar.number_input("Number of Bay - Y: ", value=1, step=1) 
        # numFloor = st.sidebar.number_input("Number of Floor: ", value=3, step=1)  
        # bayWidthX = st.sidebar.number_input("Bay Width - X: ", value=1, step=1)
        # bayWidthY = st.sidebar.number_input("Bay Width - Y: ", value=1, step=1)
        # storyHeight = st.sidebar.number_input("Story Heights - X: ", value=3.0, step=1.0)
        # E = st.sidebar.number_input("Modulus of Elasticity: ", value=28000000., step=1000000.)
        # massX = st.sidebar.number_input("Typical Mass for each joint: ", value=10, step=1)
    
    
        # set some properties
        ops.wipe()
        
        ops.model('Basic', '-ndm', 3, '-ndf', 6)
        
        # properties
        # units kN, m
        
        section_tag = 1
        secTag_1 = 1001
        beamIntTag = 1001
        secBeamTag = 2001
        
        # Define materials for nonlinear columns
        # ------------------------------------------
        # CONCRETE                  tag   f'c        ec0   ecu E
        # Core concrete (confined)
        
        # # Cover concrete (unconfined)
        # ops.uniaxialMaterial('Concrete04',2, -25000.,  -0.002,  -0.004,  28000000, 0.0, 0.0, 0,1)
        # Propiedades de los materiales
        fy = 4200000           #Fluencia del acero
        Es = 200000000.0      #Módulo de elasticidad del acero
        fc = 20000 # kg/cm2             #Resistencia a la compresión del concreto
        E  = 28000000  #Módulo de elasticidad del concreto
        G  = 0.5*E/(1+0.2)            #Módulo de corte del concreto
        
        cover = 0.04                  #Recubrimiento de vigas y columnas
        # Parametros no lineales de comportamiento del concreto
        fc1 = -fc                     #Resistencia a la compresión del concreto
        Ec1 = E                       #Módulo de elasticidad del concreto
        nuc1 = 0.2                    #Coeficiente de Poisson
        Gc1 = Ec1/(2*(1+nuc1))        #Módulo de corte del concreto
        
        # Concreto confinado
        Kfc = 1.0 # 1.3               # ratio of confined to unconfined concrete strength
        Kres = 0.2                    # ratio of residual/ultimate to maximum stress
        fpc1 = Kfc*fc1
        epsc01 = 2*fpc1/Ec1 
        fpcu1 = Kres*fpc1
        epsU1 = 5*epsc01#20
        lambda1 = 0.1
        # Concreto no confinado
        fpc2 = fc1
        epsc02 = -0.003
        fpcu2 = Kres*fpc2
        epsU2 = -0.006#-0.01
        # Propiedades de resistencia a la tracción
        ft1 = -0.14*fpc1
        ft2 = -0.14*fpc2
        Ets = ft2/0.002
        #print(E/10**8, Ets/10**8)
        
        # Concreto confinado          tag  f'c   ec0     f'cu   ecu
        # ops.uniaxialMaterial('Concrete02', 1, fpc1, epsc01, fpcu1, epsU1, lambda1, ft1, Ets)
        ops.uniaxialMaterial('Concrete02', 1, fpc1, epsc01, fpcu1, epsc01, lambda1, ft1, Ets)
        # Concreto no confinado
        ops.uniaxialMaterial('Concrete02', 2, fpc1, epsc01, fpcu1, epsc01, lambda1, ft1, Ets)
        # ops.uniaxialMaterial('Concrete02', 2, fpc2, epsc02, fpcu2, epsU2, lambda1, ft2, Ets)
        # Acero de refuerzo       tag  fy  E0  b
        ops.uniaxialMaterial('Steel02', 3, fy, Es, 0.01, 18,0.925,0.15)
        
        # fc0 = -25000.
        # fcc = -28000.
        # ecc = 0.002
        # Ec = 28000000.
        # sqrttool = sqrt(float(-fc0))
        # Ec = 5000*sqrttool
        # E = 28000000.
        # G = 11666667
        M = 0.
        
        # ops.uniaxialMaterial('Concrete04',1, int(-25000.), float(-0.002),  -0.02,  int(28000000), 0.0, 0.0, 0.1)
        
        # # Cover concrete (unconfined)
        # ops.uniaxialMaterial('Concrete04',2, -25000.,  -0.002,  -0.004,  28000000, 0.0, 0.0, 0,1)
        
        
        # # STEEL
        # # Reinforcing steel 
        # Ey = 200000000.0    # Young's modulus
        # by = 0.01
        # R0 = 15.0
        # cR1 = 0.925
        # cR2 = 0.15
        # fy = 420000
        # #                        tag  fy E0    b
        # ops.uniaxialMaterial('Steel01', 3, int(fy), Ey, by)
        
        coordTransf = "PDelta"
        coordTransf1 = "Linear"  # Linear, PDelta, Corotational
        coordTransf2 = "Linear"
        massType = "-lMass"  # -lMass, -cMass
        
        # add column element
        ops.geomTransf(coordTransf, 1, 1, 0, 0)  
        ops.geomTransf(coordTransf1, 2, 0, 0, 1)
        ops.geomTransf(coordTransf2, 3, 0, 0, 1)
        
        # ops.geomTransf(coordTransf, 1, 0, 0, 1)
        # ops.geomTransf(coordTransf1, 2, 1, 0, 0)
        # ops.geomTransf(coordTransf2, 3, 1, 0, 0)
        
        # nodeTag = 1
        
        startJointIndex = 1
        total_mass = 0
        joint_massX = df_joint_mass.U1
        joint_massY = df_joint_mass.U2
        joint_massZ = df_joint_mass.U3
        joint = df_joint.Joint
        x_coor = df_joint.XorR
        y_coor = df_joint.Y
        z_coor = df_joint.Z
        deneme = []
        while startJointIndex <= total_joint_rows:
            ops.node(int(joint[startJointIndex]), x_coor[startJointIndex], y_coor[startJointIndex], z_coor[startJointIndex])
            ops.mass(int(joint[startJointIndex]), joint_massX[startJointIndex], joint_massY[startJointIndex], joint_massZ[startJointIndex], 1.0e-10, 1.0e-10, 1.0e-10)
            total_mass = total_mass + joint_massX[startJointIndex] 
            startJointIndex +=1
        
        
        startRestrainIndex = 0
        restrain = df_restrain.Joint
        while startRestrainIndex <= total_restrain_rows-1:
            ops.fix(restrain[startRestrainIndex], 1, 1, 1, 1, 1, 1)
            startRestrainIndex += 1
        
        frame = df_frame.Frame
        joint_I = df_frame.JointI
        joint_J = df_frame.JointJ
        frame_section = df_frame_section.AnalSect
        startFrameIndex = 0   
        Area = df_section_properties["Area"].tolist()
        TorsConst = df_section_properties["TorsConst"].tolist()
        I33 = df_section_properties["I33"].tolist()
        I22 = df_section_properties["I22"].tolist()
        t3 = df_section_properties.t3
        t2 = df_section_properties.t2
        section_cover = df_section_rebar.Cover
        nol = df_section_rebar.NumBars2Dir
        number_of_top = df_section_rebar.NumBars3Dir
        number_of_bottom = df_section_rebar.NumBars3Dir
        long_bar = df_section_rebar.BarSizeL
        width_total = []
        depth_total = []
        section_type1 = []
        frame_list = []
        beam_depth = []
        beam_width = []
        while startFrameIndex <= total_frame_rows-1:
            frame1 = frame[startFrameIndex]
            frame_index = df_frame["Frame"].tolist().index(frame[startFrameIndex])
            analysis_section = frame_section[frame_index]
            analysis_index = df_section_properties["SectionName"].tolist().index(frame_section[frame_index])
        
            joint1 = joint_I[startFrameIndex]
            joint2 = joint_J[startFrameIndex]
            jointI_index = df_joint["Joint"].tolist().index(joint_I[startFrameIndex])
            jointJ_index = df_joint["Joint"].tolist().index(joint_J[startFrameIndex])
            z_coordinate_I = z_coor[jointI_index+1]
            z_coordinate_J = z_coor[jointJ_index+1]
            y_coordinate_I = y_coor[jointI_index+1]
            y_coordinate_J = y_coor[jointJ_index+1]
            x_coordinate_I = x_coor[jointI_index+1]
            x_coordinate_J = x_coor[jointJ_index+1]
            
        
            if z_coordinate_I == z_coordinate_J:
                
                section_type1.append("Beam")
                
                pi = 3.141593;
                dia1 = long_bar[analysis_index]/1000
                As = dia1**2*pi/4;     # area of no. 7 bars
                
                width = t3[analysis_index]
                depth = t2[analysis_index]  
                width_total.append(width)
                depth_total.append(depth)
                frame_list.append(frame[startFrameIndex])
                cover = section_cover[analysis_index]
                number_of_layer = nol[analysis_index]
                n_top = number_of_top[analysis_index]
                n_bot = number_of_bottom[analysis_index]
                n_int = 2
                
                b1 = width/2 - cover
                b2 = (width/2 - cover)*-1
                h1 = depth/2 - cover
                h2 = (depth/2 - cover)*-1
                k_1 = 1/3-0.21*width/depth*(1-(width/depth)**4/12)
                Jc = k_1*width**3*depth
                
                # some variables derived from the parameters
                y1 = depth/2.0
                z1 = width/2.0
                total_y = depth - 2*cover
                total_y_layer = total_y/(number_of_layer-1)
                total_y_layer_step = total_y/(number_of_layer-1)
        
                
                ops.section('Elastic', secTag_1, E, Area[analysis_index], I33[analysis_index], I22[analysis_index], G, TorsConst[analysis_index])
        
                ops.section('Fiber', section_tag, '-GJ', G*Jc)
                
                # Create the concrete core fibers
                ops.patch('rect',2,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover)
                        
                # Create the concrete cover fibers (top, bottom, left, right)
                ops.patch('rect',2,50,1 ,-y1, z1-cover, y1, z1)
                ops.patch('rect',2,50,1 ,-y1, -z1, y1, cover-z1)
                ops.patch('rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover)
                ops.patch('rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover)
                
                top = ['layer','straight', 3, n_top, As, y1-cover-dia1, cover-z1+dia1, y1-cover-dia1, z1-cover-dia1]
                bottom = ['layer','straight', 3, n_bot, As, cover-y1+dia1, cover-z1+dia1, cover-y1+dia1, z1-cover-dia1]
                
                fib_sec_2 = [['section', 'Fiber', 1],
                ['patch', 'rect',2,50,1 ,-y1, z1-cover, y1, z1],
                ['patch', 'rect',2,50,1 ,-y1, -z1, y1, cover-z1],
                ['patch', 'rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover],
                ['patch', 'rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover],
                ['patch', 'rect',1,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover],
                top,
                bottom]
                
                ops.layer('straight', 3, n_top, As, y1-cover, cover-z1, y1-cover, z1-cover)
                ops.layer('straight', 3, n_bot, As, cover-y1, cover-z1, cover-y1, z1-cover)
                
                total_int_layer = number_of_layer-2
                int_layer = 1
                
                ops.beamIntegration("Lobatto", beamIntTag,section_tag,5)
                
                while int_layer <= total_int_layer:
                
                    ops.layer('straight', 3, n_int, As, y1-cover-total_y_layer, cover-z1+dia1, y1-cover-total_y_layer, z1-cover-dia1)
                    int_layer_def = ['layer','straight', 3, n_int, As, y1-cover-total_y_layer, cover-z1+dia1, y1-cover-total_y_layer, z1-cover-dia1]
                    fib_sec_2.append(int_layer_def)
                    total_y_layer = total_y_layer + total_y_layer_step
                    int_layer = int_layer +1
                    
                matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
                # opsv.plot_fiber_section(fib_sec_2, matcolor=matcolor)
                plt.axis('equal')    
                numIntgrPts = 5
                
                if y_coordinate_I == y_coordinate_J:
                    if switch == 1:
                        ops.element('forceBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), 2, beamIntTag)
                    elif switch == 2:
                        ops.element('nonlinearBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), numIntgrPts, section_tag, 2, '-integration', 'Lobatto')
                    elif switch == 3:
                        ops.element('elasticBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), Area[analysis_index], E, G, TorsConst[analysis_index], I33[analysis_index], I22[analysis_index], 2, '-mass', M, massType)
                else:
                    if switch == 1:
                        ops.element('forceBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), 3, beamIntTag)
                    elif switch == 2:
                        ops.element('nonlinearBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), numIntgrPts, section_tag, 3, '-integration', 'Lobatto')
                    elif switch == 3:
                        ops.element('elasticBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), Area[analysis_index], E, G, TorsConst[analysis_index], I33[analysis_index], I22[analysis_index], 3, '-mass', M, massType)
            else:
                
                pi = 3.141593;
                dia1 = long_bar[analysis_index]/1000
                As = dia1**2*pi/4;     # area of no. 7 bars
                section_type1.append("Column")
                width = t2[analysis_index]
                depth = t3[analysis_index]
                width_total.append(width)
                depth_total.append(depth)
                frame_list.append(frame[startFrameIndex])
                cover = section_cover[analysis_index]
                number_of_layer = nol[analysis_index]
                n_top = number_of_top[analysis_index]
                n_bot = number_of_bottom[analysis_index]
                n_int = 2
                
                b1 = width/2 - cover
                b2 = (width/2 - cover)*-1
                h1 = depth/2 - cover
                h2 = (depth/2 - cover)*-1
                k_1 = 1/3-0.21*width/depth*(1-(width/depth)**4/12)
                Jc = k_1*width**3*depth
                
                # some variables derived from the parameters
                y1 = depth/2.0
                z1 = width/2.0
                total_y = depth - 2*cover
                total_y_layer = total_y/(number_of_layer-1)
                total_y_layer_step = total_y/(number_of_layer-1)
                
                ops.section('Elastic', secTag_1, E, Area[analysis_index], I33[analysis_index], I22[analysis_index], G, TorsConst[analysis_index])
                        
                ops.section('Fiber', section_tag, '-GJ', G*Jc)
                
                # Create the concrete core fibers
                ops.patch('rect',1,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover)
                
                
                # Create the concrete cover fibers (top, bottom, left, right)
                ops.patch('rect',2,50,1 ,-y1, z1-cover, y1, z1)
                ops.patch('rect',2,50,1 ,-y1, -z1, y1, cover-z1)
                ops.patch('rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover)
                ops.patch('rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover)
                
                top = ['layer','straight', 3, n_top, As, y1-cover-dia1, cover-z1+dia1, y1-cover-dia1, z1-cover-dia1]
                bottom = ['layer','straight', 3, n_bot, As, cover-y1+dia1, cover-z1+dia1, cover-y1+dia1, z1-cover-dia1]
                
                fib_sec_2 = [['section', 'Fiber', 1],
                ['patch', 'rect',2,50,1 ,-y1, z1-cover, y1, z1],
                ['patch', 'rect',2,50,1 ,-y1, -z1, y1, cover-z1],
                ['patch', 'rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover],
                ['patch', 'rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover],
                ['patch', 'rect',1,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover],
                top,
                bottom]
                
                ops.layer('straight', 3, n_top, As, y1-cover, cover-z1, y1-cover, z1-cover)
                ops.layer('straight', 3, n_bot, As, cover-y1, cover-z1, cover-y1, z1-cover)
                
                total_int_layer = number_of_layer-2
                int_layer = 1
                
                ops.beamIntegration("Lobatto", beamIntTag,section_tag,5)
                
                while int_layer <= total_int_layer:
                
                    ops.layer('straight', 3, n_int, As, y1-cover-total_y_layer, cover-z1+dia1, y1-cover-total_y_layer, z1-cover-dia1)
                    int_layer_def = ['layer','straight', 3, n_int, As, y1-cover-total_y_layer, cover-z1+dia1, y1-cover-total_y_layer, z1-cover-dia1]
                    fib_sec_2.append(int_layer_def)
                    total_y_layer = total_y_layer + total_y_layer_step
                    int_layer = int_layer +1
                    
                matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
                # opsv.plot_fiber_section(fib_sec_2, matcolor=matcolor)
                plt.axis('equal')  
                
                numIntgrPts = 8
                if switch == 1:
                    ops.element('forceBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), 1, beamIntTag)
                elif switch == 2:
                    ops.element('nonlinearBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), numIntgrPts, section_tag, 1, '-integration', 'Lobatto')
                elif switch == 3:
                    ops.element('elasticBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), Area[analysis_index], E, G, TorsConst[analysis_index], I33[analysis_index], I22[analysis_index], 1, '-mass', M, massType)
            startFrameIndex +=1
            section_tag += 1
            secTag_1 += 1
            beamIntTag += 1
            secBeamTag += 1
        
        # calculate eigenvalues & print results
        numEigen = 3
        eigenValues = ops.eigen(numEigen)
        PI = 2 * asin(1.0)
        
        period_list = []
        for i in range(0, numEigen):
            lamb = eigenValues[i]
            period = 2 * PI / sqrt(lamb)
            period_list.append(period)
        
                
        ###################################
        #### Display the active model with node tags only
        
        ####  Display specific mode shape with scale factor of 300 using the active model
        # fig_wi_he2 = 30., 20.
        # opsplt.plot_modeshape(2, 20, Model="3DFrame")
        
        
        ###################################
        # To save the analysis output for deformed shape, use createODB command before running the analysis
        # The following command saves the model data, and output for gravity analysis and the first 3 modes 
        # in a folder "3DFrame_ODB"
        
        # opsplt.createODB("3DFrame", "Gravity", Nmodes=3)
        
        
        # # # Define Static Analysis
        # ops.timeSeries('Linear', 1)
        # ops.pattern('Plain', 1, 1)
        # ops.load(total_joint_rows, 3, 0, 0, 0, 0, 0)
        # ops.constraints('Transformation')
        # ops.numberer('RCM')
        # ops.system('BandGeneral')
        # ops.test('NormDispIncr', 1.0e-6, 6, 2)
        # ops.algorithm('Linear')
        # ops.integrator('LoadControl', 1)
        # ops.analysis('Static')
        
        # # # Run Analysis
        # ops.analyze(10)
        
        # for i in range(0, numEigen):
        #     lamb = eigenValues[i]
        #     period = 2 * PI / sqrt(lamb)
        
        # # IMPORTANT: Make sure to issue a wipe() command to close all the recorders. Not issuing a wipe() command
        # # ... can cause errors in the plot_deformedshape() command.
        
        # ops.wipe()
        
        # ####################################
        # ### Now plot mode shape 2 with scale factor of 300 and the deformed shape using the recorded output data
        
        # opsplt.plot_modeshape(2, 300, Model="3DFrame")
        # opsplt.plot_deformedshape(Model="3DFrame", LoadCase="Gravity")
        
        # st.success("Analysis Compeleted!")
        # st.info("First Period of System is: " + str(format(period, ".2f")) + " sec")
        # st.markdown("First Period of System is:" + str(format(period, ".2f")) + " sec")
        
        # fig_wi_he = 30., 20.
        # ele_shapes = []
        # startFrameIndex = 0
        # while startFrameIndex <= total_frame_rows-1:
        #     x = ['rect', [width_total[startFrameIndex], depth_total[startFrameIndex]]]
        #     ele_shapes.append(x)   
        #     startFrameIndex = startFrameIndex +1
            
        # fruit_dictionary = dict(zip(frame_list, ele_shapes))
        # deneme3 = opsv.plot_extruded_shapes_3d(fruit_dictionary, fig_wi_he=fig_wi_he)
        
        def str_system():
            co1, co2 = st.columns(2)
            with co1:
                fig_wi_he = 30., 20.
                ele_shapes = []
                startFrameIndex = 0
                while startFrameIndex <= total_frame_rows-1:
                    x = ['rect', [width_total[startFrameIndex], depth_total[startFrameIndex]]]
                    ele_shapes.append(x)   
                    startFrameIndex = startFrameIndex +1
                    
                fruit_dictionary = dict(zip(frame_list, ele_shapes))
                deneme3 = opsv.plot_extruded_shapes_3d(fruit_dictionary, fig_wi_he=fig_wi_he)
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(deneme3)
            with co2:
                opsplt.plot_model("nodes")
                nodes=opsplt.createODB("3DFrame", "Gravity", Nmodes=3)
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(nodes)
            
                
        def mode_shapes():
            co1, co2, co3 = st.columns(3)
            with co1:
                opsplt.plot_modeshape(1, 50)
                mode1=opsplt.createODB("3DFrame", "Gravity", Nmodes=3)
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(mode1)
            with co2:    
                ####  Display specific mode shape with scale factor of 300 using the active model
                opsplt.plot_modeshape(2, 50)
                mode2=opsplt.createODB("3DFrame", "Gravity", Nmodes=3)
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(mode2)
            with co3:   
                ####  Display specific mode shape with scale factor of 300 using the active model
                opsplt.plot_modeshape(3, 50)
                mode3=opsplt.createODB("3DFrame", "Gravity", Nmodes=3)
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(mode3)
        
        def period_fun():
            first_period = period_list[0]
            second_period = period_list[1]
            third_period = period_list[2]
            return first_period, second_period, third_period
        
        first_period, second_period, third_period = period_fun()
        
        st.subheader("Exact Location of the Structure")
        
        # This part uses for the get data from Excel sheet about the spectral variables.
        
        my_sheet_main = 'Sayfa1' # change it to your sheet name
        main_file_name = 'parametre_UPD.xlsx' # change it to the name of your excel file
        df_main = read_excel(main_file_name, sheet_name = my_sheet_main, engine='openpyxl')
        # print(df_main.head()) # shows headers with top 5 rows
        # print(df_main.info())
        
        my_sheet = 'Sayfa1' # change it to your sheet name
        file_name = 'unique_lat_lon.xlsx' # change it to the name of your excel file
        df = read_excel(file_name, sheet_name = my_sheet, engine='openpyxl')
        # print(df.head()) # shows headers with top 5 rows
        # print(df.info())
        
        r_d = 'Sayfa1' # change it to your sheet name
        file_name = 'r_d.xlsx' # change it to the name of your excel file
        df_r_d = read_excel(file_name, sheet_name = r_d, engine='openpyxl')
        # print(df.head()) # shows headers with top 5 rows
        # print(df.info())
        
        longitude = df["Boylam"].to_frame()
        latitude = df["Enlem"].to_frame()
        
        # Let's start the streamlit commands for inputs
        
        
        st.sidebar.header("Structure Location")
        x = st.sidebar.number_input("Longitude (Range: 26.00 - 45.00)",value=29.01, step=0.1)
        y = st.sidebar.number_input("Latitude: (Range: 36.00 - 42.00)",value=41.10, step=0.1)
        
        raw_data = {'lat': [y],
        'lon': [x]}
        df = pd.DataFrame(raw_data, columns = ['lat', 'lon'])
        
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=y,
                longitude=x,
                zoom=11,
                pitch=50,
                ),
            layers=[
                # pdk.Layer(
                #     'HexagonLayer',
                #     data=df,
                #     get_position='[lon, lat]',
                #     radius=200,
                #     elevation_scale=4,
                #     elevation_range=[0, 1000],
                #     pickable=True,
                #     extruded=True,
                #     ),
                pdk.Layer(
                    'ScatterplotLayer',
                    data=df,
                    get_position='[lon, lat]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=1000,
                    ),
                ],
            ))
        
        """
        
        Please be sure the coordinates of the structure by the help of map. 
        This map will show your location that you selected from left side of the screen.
        
        """
        # Displaying Latitude and Longitude
        print("Latitude: ", str(y))
        print("Longitude: ", str(x))
         
        # Get location with geocode
        location = geolocator.geocode(str(y)+","+str(x))
        
        st.success("Adress: " + str(location))
        st.success("Longitude: " + str(x) + " & Latitude: " + str(y))
        
        
        st.subheader("Type of Structure")
        
        """
        Please choose structure' type, ductility and supporting system from the list.
        
        Tables that shared below will be updated according to your choices.
        
        """
        
        """
        According to TBEC 2018, to draw SaR/SaE - Period(T) graphs, you should use
        same parameters about the structural system. 
        
        These parameters can be seen below:
            
            1- BTS (Building Supporting System)
            2- R (Response Modification Factor)
            3- D (System Overstrength Factor)
        
        """
        
        str_system()
        
        
        
        structure_type = st.selectbox("Type of Structure: ", {"Steel", "Concrete"})
        ductility = st.selectbox("Ductility of Structure: ", {"High", "Moderate", "Low"})
        if structure_type == "Steel":
            if ductility == "High":
                df_r_d_sub = df_r_d[13:19]
                st.write(df_r_d_sub[["BTS","R","D"]])
                structure_category = st.selectbox("Category: ", {"C11", "C12","C13","C14","C15","C16"})
        
            elif ductility == "Moderate":
                df_r_d_sub = df_r_d[19:22]
                st.write(df_r_d_sub[["BTS","R","D"]])
                structure_category = st.selectbox("Category: ", {"C21", "C22"})
            else:
                df_r_d_sub = df_r_d[22:24]
                st.write(df_r_d_sub[["BTS","R","D"]])
                structure_category = st.selectbox("Category: ", {"C31", "C32","C33"})
        elif structure_type == "Concrete":
            if ductility == "High":
                df_r_d_sub = df_r_d[0:6]
                st.write(df_r_d_sub[["BTS","R","D"]])
                structure_category = st.selectbox("Category: ", {"A11", "A12", "A13", "A14", "A15", "A16"})
            elif ductility == "Moderate":
                df_r_d_sub = df_r_d[6:10]
                st.write(df_r_d_sub[["BTS","R","D"]])
                structure_category = st.selectbox("Category: ", {"A21", "A22", "A23","A24"})
            else:
                df_r_d_sub = df_r_d[10:13]
                st.write(df_r_d_sub[["BTS","R","D"]])
                structure_category = st.selectbox("Category: ", {"A31", "A32","A33"})
        
        str_cat_final = df_r_d.loc[df_r_d['BTS']==structure_category]
        R = str_cat_final['R'].iloc[0]
        D = str_cat_final['D'].iloc[0]
        ToSS = str_cat_final['BTS'].iloc[0]
        
        # st.markdown("Type of Structural System: " + str(ToSS))
        # st.markdown("R: " + str(R))
        # st.markdown("D: " + str(D))
        
        I_var = [1.0,1.2,1.5]
        
        df_I = pd.DataFrame(list(I_var))
        df_I.columns = ['I']
        st.write(df_I)
        # R = st.sidebar.selectbox("Seismic Response Modification Factor - R: ", {1,2,3,4,5,6,7,8})
        # D = st.sidebar.selectbox("Overstrength Coefficient - D: ",{1, 1.5, 2, 2.5, 3})
        I = st.selectbox("Building Importance Factor - I : ", {1, 1.2, 1.5})
        
        soilType = st.selectbox("Soil Type: ", {"ZA","ZB","ZC","ZD","ZE","ZF"})
        
        first_period, second_period, third_period = period_fun()
        period_define = st.selectbox("Period: ", {"From Analysis", "User Defined"})
        
        if period_define == "From Analysis":
            T_x = st.selectbox("Period - X Direction: ", {"1st Mode", "2nd Mode", "3rd Mode"})
            if T_x == "1st Mode":
                T_x = first_period
            elif T_x == "2nd Mode":
                T_x = second_period
            elif T_x == "3rd Mode":
                T_x = third_period
            mode_shapes()
        elif period_define == "User Defined":
            T_x = st.number_input("Period - X Direction:: ",value=0.20, step=0.1)
        total_weight = st.number_input("Total Weight (kN): ",value=100, step=1)
        
        # print(latitude.iloc[(latitude['Enlem']-y).abs().argsort()[:2]])
        # print(longitude.iloc[(longitude['Boylam']-x).abs().argsort()[:2]])
        
        df_sort_y = latitude.iloc[(latitude['Enlem']-y).abs().argsort()[:2]]
        # print(df_sort_y.index.tolist())
        df_sort_x = longitude.iloc[(longitude['Boylam']-x).abs().argsort()[:2]]
        # print(df_sort_x.index.tolist())
        
        numbers_y = df_sort_y['Enlem'].tolist()
        numbers_x = df_sort_x['Boylam'].tolist()
        
        
        x1 = min(numbers_x)
        x2 = max(numbers_x)
        y1 = min(numbers_y)
        y2 = max(numbers_y)  
        
        x_f = (x-x1)/(x2-x1)
        y_f = (y-y1)/(y2-y1)
        
        var1_look = str(x1)+str(y1)
        var2_look = str(x2)+str(y1)
        var3_look = str(x1)+str(y2)
        var4_look = str(x2)+str(y2)
        
        variables_list = [var1_look, var2_look, var3_look,var4_look]
        df_variables = pd.DataFrame(variables_list)
        df_variables.columns = ['Merged']
        
        var1_final = df_main.loc[df_main['Merged']==var1_look]
        var2_final = df_main.loc[df_main['Merged']==var2_look]
        var3_final = df_main.loc[df_main['Merged']==var3_look]
        var4_final = df_main.loc[df_main['Merged']==var4_look]
        
        # for DD2
        var1_s1= var1_final['S12'].iloc[0]
        var2_s1= var2_final['S12'].iloc[0]
        var3_s1= var3_final['S12'].iloc[0]
        var4_s1= var4_final['S12'].iloc[0]
        var1_ss= var1_final['SS2'].iloc[0]
        var2_ss= var2_final['SS2'].iloc[0]
        var3_ss= var3_final['SS2'].iloc[0]
        var4_ss= var4_final['SS2'].iloc[0]
        
        var1_var2_s1 = var1_s1+x_f*(var2_s1-var1_s1)
        var3_var4_s1 =var3_s1+x_f*(var4_s1-var3_s1)
        var1_var2_ss = var1_ss+x_f*(var2_ss-var1_ss)
        var3_var4_ss =var3_ss+x_f*(var4_ss-var3_ss)
        
        s1 = var1_var2_s1 + y_f*(var3_var4_s1 - var1_var2_s1)
        ss = var1_var2_ss + y_f*(var3_var4_ss - var1_var2_ss)
        
        # for DD3
        var1_s1_DD3= var1_final['S13'].iloc[0]
        var2_s1_DD3= var2_final['S13'].iloc[0]
        var3_s1_DD3= var3_final['S13'].iloc[0]
        var4_s1_DD3= var4_final['S13'].iloc[0]
        var1_ss_DD3= var1_final['SS3'].iloc[0]
        var2_ss_DD3= var2_final['SS3'].iloc[0]
        var3_ss_DD3= var3_final['SS3'].iloc[0]
        var4_ss_DD3= var4_final['SS3'].iloc[0]
        
        var1_var2_s1_DD3 = var1_s1_DD3+x_f*(var2_s1_DD3-var1_s1_DD3)
        var3_var4_s1_DD3 =var3_s1_DD3+x_f*(var4_s1_DD3-var3_s1_DD3)
        var1_var2_ss_DD3 = var1_ss_DD3+x_f*(var2_ss_DD3-var1_ss_DD3)
        var3_var4_ss_DD3 =var3_ss_DD3+x_f*(var4_ss_DD3-var3_ss_DD3)
        
        s1_DD3 = var1_var2_s1_DD3 + y_f*(var3_var4_s1_DD3 - var1_var2_s1_DD3)
        ss_DD3 = var1_var2_ss_DD3 + y_f*(var3_var4_ss_DD3 - var1_var2_ss_DD3)
                                    
        # find center and radius
        sDs, sD1, Fs, F1 = soilType_func(soilType, ss, s1)
        print(Fs)
        sDs_DD2 = sDs
        sD1_DD2 = sD1
        Fs1_DD2 = Fs
        F1_DD2 = F1
        sDs, sD1, Fs, F1 = soilType_func(soilType, ss_DD3, s1_DD3)
        sDs_DD3 = sDs
        sD1_DD3 = sD1
        Fs1_DD3 = Fs
        F1_DD3 = F1
        
        tSpectrum = []
        for i in arange(0.0, 8.0, 0.001):
            tSpectrum.append(format(i, ".3f"))
        
        tA = (0.2*sD1_DD2/sDs_DD2)
        tB = (sD1_DD2/sDs_DD2)
        tA_DD3 = (0.2*sD1_DD3/sDs_DD3)
        tB_DD3 = (sD1_DD3/sDs_DD3)
        tL = 6.0
        length_T = len(tSpectrum)
        
        sAec = []
        sAec_DD3 = []
        tRs = []
        tRs_DD3 = []
        
        i = 0
        while i < length_T:
            if float(tSpectrum[i]) >= 0.0 and float(tSpectrum[i]) < tA:
                sAe = (0.4+0.6*float(tSpectrum[i])/tA)*sDs_DD2
                tRs.append(format(sAe, ".3f"))
            elif float(tSpectrum[i]) >= tA and float(tSpectrum[i]) < tB:
                sAe = sDs_DD2
                tRs.append(format(sAe, ".3f"))
            elif float(tSpectrum[i]) >= tB and float(tSpectrum[i]) < tL:
                sAe = sD1_DD2/float(tSpectrum[i])
                tRs.append(format(sAe, ".3f"))
            elif float(tSpectrum[i]) >= tL:
                sAe = sD1_DD2*tL / float(tSpectrum[i])**2
                tRs.append(format(sAe, ".3f"))
        
            i += 1
            
        i = 0
        while i < length_T:
            if float(tSpectrum[i]) >= 0.0 and float(tSpectrum[i]) < tA_DD3:
                sAe_DD3 = (0.4+0.6*float(tSpectrum[i])/tA_DD3)*sDs_DD3
                tRs_DD3.append(format(sAe_DD3, ".3f"))
            elif float(tSpectrum[i]) >= tA_DD3 and float(tSpectrum[i]) < tB_DD3:
                sAe_DD3 = sDs_DD3
                tRs_DD3.append(format(sAe_DD3, ".3f"))
            elif float(tSpectrum[i]) >= tB_DD3 and float(tSpectrum[i]) < tL:
                sAe_DD3 = sD1_DD3/float(tSpectrum[i])
                tRs_DD3.append(format(sAe_DD3, ".3f"))
            elif float(tSpectrum[i]) >= tL:
                sAe_DD3 = sD1_DD3*tL / float(tSpectrum[i])**2
                tRs_DD3.append(format(sAe_DD3, ".3f"))
        
            i += 1
            
        df = pd.DataFrame(list(zip(tSpectrum, tRs)), columns =['Period', 'Sae'], dtype = float)
        df_DD3 = pd.DataFrame(list(zip(tSpectrum, tRs_DD3)), columns =['Period', 'Sae_DD3'], dtype = float)
        # df.plot(x ='Period', y='Sae', kind = 'line')
        # plt.show()
        
        
        rA = []
        
        i = 0
        while i < length_T:
            if float(tSpectrum[i]) >= tB:
                rAT = R/I
                rA.append(format(rAT, ".3f"))
            elif float(tSpectrum[i]) <= tB:
                rAT = D+(R/I-D)*float(tSpectrum[i])/tB
                rA.append(format(rAT, ".3f"))
            i += 1
        
        df["R"] = rA
        df["R"] = df["R"].astype(float)
        
        # print(df.head())
        
        period_spec = df["Period"].to_frame()
        # print(period_spec.iloc[(period_spec['Period']-T).abs().argsort()[:2]])
        df_period = period_spec.iloc[(period_spec['Period']-T_x).abs().argsort()[:2]]
        numbers_period = df_period['Period'].tolist()
        T = min(numbers_period)
        
        df["Sar"] = df["Sae"]/df["R"]
        # print(df.head())
        
        t_final = df.loc[df['Period']==T]
        t_final_DD3 = df_DD3.loc[df_DD3['Period']==T]
        Sar = t_final['Sar'].iloc[0]
        Sae = t_final['Sae'].iloc[0]
        Sae_DD3 = t_final_DD3['Sae_DD3'].iloc[0]
        exact_point = {'Period': [T], 'Sar': [Sar]}
        design_point = pd.DataFrame(exact_point)
        
        
        # print("Design Spectral Acceleration is: " + str(format(Sar, ".3f")) + "g")
        
        st.sidebar.header("General Information of the Structural System")
        
        st.sidebar.markdown("Type of the Structure: " + structure_type)
        st.sidebar.markdown("Ductility: " + ductility )
        st.sidebar.markdown("Supporting System Category: " + structure_category )
        st.sidebar.markdown("Building Importance Factor: " + str(I))
        st.sidebar.markdown("Soil Type: " + soilType )
        st.sidebar.markdown("Location: Lon: " + str(x) + " & Lat: " + str(y))
        
        st.sidebar.markdown("Period of Structure: " + str(format(T, ".2f")) + "s")
           
            
        # st.sidebar.markdown("SaR: " + str(format(Sar, ".3f")) + "g")
        # st.sidebar.markdown("SaE: " + str(format(Sae, ".3f")) + "g")
        st.sidebar.success("SaE: " + str(format(Sae, ".3f")) + "g")
        st.sidebar.success("SaR: " + str(format(Sar, ".3f")) + "g")
        st.sidebar.success("Base Reaction: " + str(format(Sar*total_weight,".2f") + "kN"))
        # p = figure(
        #     title="Design Spectrum",
        #     x_axis_label="Period",
        #     y_axis_label="Spectral Acceleration",
        #     match_aspect=True,
        #     tools="pan,reset,save,wheel_zoom",
        # )
        
        # t_period = df["Period"].to_frame()
        # Sar = df["Sar"].to_frame()
        # p.line(t_period, Sar, color="#1f77b4", line_width=3, line_alpha=0.6)
        
        
        # p.xaxis.fixed_location = 0
        # p.yaxis.fixed_location = 0
        
        # st.bokeh_chart(p)
        
        import matplotlib.pyplot as plt
        
        
        # fig_saE = plt.figure()
        # ax = fig_saE.add_subplot(1,1,1)
        
        # ax.scatter(df["Period"], df["Sae"],color='r',linewidth=1)
        # # ax = design_point.plot(x ='Period', y='Sar', kind = 'scatter',c = "red", label = "Design Spectrum")
        
        # ax.set_xlabel("Period (T)")
        # ax.set_ylabel("SaE (g)")
        
        
        # df.plot(ax=ax, x ='Period', y='Sar', kind = 'line', label = "Design Point")
        
        # st.write(fig_saE)
        
        
        
        # fig_saR = plt.figure()
        # ax = fig_saR.add_subplot(1,1,1)
        
        # ax.scatter(df["Period"], df["Sar"],color='r',linewidth=1)
        # # ax = design_point.plot(x ='Period', y='Sar', kind = 'scatter',c = "red", label = "Design Spectrum")
        
        # ax.set_xlabel("Period (T)")
        # ax.set_ylabel("SaR (g)")
        
        #Line Chart
        as_list = df["Period"].tolist()
        
        df.index = as_list
        
        df_Sar = df['Sar']
        df_Sae = df['Sae']
        
        """
        Elastic Spectral Acceleration - Period Graph
        
        """
        st.line_chart(df_Sae)
        
        """
        Design Spectral Acceleration - Period Graph
        
        """
        
        st.line_chart(df_Sar)
        
        def convert_design_df(df_Sar):
           return df_Sar.to_csv().encode('utf-8')
        def convert_elastic_df(df_Sae):
           return df_Sae.to_csv().encode('utf-8')
        
        csv_design = convert_design_df(df_Sar)
        csv_elastic = convert_elastic_df(df_Sae)
        
        st.download_button(
           "Design Spectrum - Press to Download",
           csv_design,
           "design_spectrum.csv",
           "text/csv",
           key='download-csv'
        )
        
        st.download_button(
           "Elastic Spectrum - Press to Download",
           csv_elastic,
           "elastic_spectrum.csv",
           "text/csv",
           key='download-csv'
        )
        
        # df.plot(ax=ax, x ='Period', y='Sar', kind = 'line', label = "Design Point")
        
        # st.write(fig_saR)
        
        # Displacement Kontrol
        
        """
        Displacement Limit and Control
        
        """
        
        number_of_story = st.number_input("Number of Story - N: ",value=1, step=1)
        height = st.number_input("Total Height of the Structure - H: ",value=3, step=1)
        delta_i = st.number_input("Maximum Inelastic Displacement under the Earthquake: ",value=0.01, step=0.01)
        
        if structure_type == "Steel":
            kappa = 0.5
        elif structure_type == "Concrete":
            kappa = 1.0
        lambda_disp = Sae_DD3/Sae
        delta_design = R*delta_i/I
        delta_max = delta_design
        ratio_ = lambda_disp*(delta_max/height)
        
        wall_contact = st.selectbox("Wall Contact: ", {"Rigid","Flexible"})
        
        
        if wall_contact == "Rigid":
            if number_of_story == 1 and structure_type == "Steel":
                limit_ratio = 0.008*kappa*1.5
            elif number_of_story == 1 and structure_type == "Concrete":
                limit_ratio = 0.008*kappa
        elif wall_contact == "Flexible":
            if number_of_story != 1 and structure_type == "Steel":
                limit_ratio = 0.016*kappa*1.5
            elif number_of_story != 1 and structure_type == "Concrete":
                limit_ratio = 0.016*kappa
        
        if limit_ratio < ratio_ :
            st.info("Displacement Ratio: " + str(format(ratio_ , ".3f")))
            st.info("Displacement Limit Ratio: " + str(limit_ratio))
            st.success("Displacement Check: NOT OK!")
        elif limit_ratio > ratio_ :
            st.info("Displacement Ratio: " + str(format(ratio_ , ".3f")))
            st.info("Displacement Limit Ratio: " + str(limit_ratio))
            st.success("Displacement Check: OK!")
        
        hide_menu_style = """
                <style>
                #MainMenu {visibility: hidden;}
                </style>
                """
        st.markdown(hide_menu_style, unsafe_allow_html=True)
    
    else:
        print("Please choose a file...")
elif design_type == "Nonlinear Design":
    st.info("This feature will be added later...")