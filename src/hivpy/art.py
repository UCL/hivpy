from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import importlib.resources
import operator as op
from enum import Enum, IntEnum

import numpy as np
import pandas as pd

import hivpy.column_names as col

from . import output
from .art_data import ARTData
from .common import OR, AND, COND, SexType, date, opposite_sex, rng, timedelta


class HivMonitoringStrategy(Enum):
    # strategy for monitoring HIV positive people naive to ART 1: presence of tb or who4
    presence_tb_who4 = 1
    # strategy for monitoring HIV positive people naive to ART 2: cd4 6 monthly + presence of tb or who4
    cd4_6_monthly = 2

class ArtInitiationStrategy(Enum):
    # all with who4
    all_who4 = 1
    # all with tb or who4
    all_tb_who4 = 2
    # all with hiv diagnosed
    all_hiv_diagnosed = 3
    # cd4 < 200 or who4
    cd4_lt_200_who4 = 4
    # cd4 < 200 or tb or who4
    cd4_lt_200_tb_who4 = 5
    # cd4 < 350 or who4
    cd4_lt_350_who4 = 6
    # cd4 < 350 and ART immediately to pregnant women
    cd4_lt_350_pregnant = 9
    # cd4 < 500 and ART immediately to pregnant women
    cd4_lt_500_pregnant = 10

class ArtMonitoringStrategy(Enum):
    # 1. Clinical monitoring alone
    only_clinical = 1
    # 2. Clinical monitoring with single VL confirmation
    clinical_single_vl = 2
    # 3. Clinical monitoring with VL confirmation
    clinical_vl_confirm = 3
    # 7. Clinical monitoring with CD4 confirmation
    clinical_cd4_confirm = 7
    # 8. CD4 monitoring (6 mthly) alone
    only_cd4_monitor = 8
    # 9. CD4 count monitoring (6 mthly) with single VL confirmation
    cd4_monitor_single_vl = 9
    # 10. CD4 monitoring (6 mthly) with VL confirmation
    cd4_monitor_vl_confirm = 10
    # 150. Viral load monitoring (6m, 12m, annual) - WHO
    vl_monitor_who = 150
    # 152. As above with 2 yearly viral load monitoring
    vl_monitor_biannual = 152
    # 153. Viral load monitoring (6m, annual) no confirmation
    vl_monitor_no_confirm = 153
    # 1500. Viral load monitoring (6m, 12m, annual) + adh > 0.8 based on tdf level test;
    vl_monitor_tdf_test = 1500
    # 1700. Monitoring people on len/cab 
    on_len_cab = 1700

class VmFormat(Enum):
    # vm_format=1  plasma  lab
    plasma_lab = 1
    # vm_format=2  whb     lab
    whb_lab = 2 
    # vm_format=3  plasma  poc
    plasma_poc = 3 
    # vm_format=4  whb     poc
    whb_poc = 4

class ARV(Enum):
    lamivudine = 1
    zidovudine = 2
    tenofovir = 3
    nevirapine = 4
    efavirenz = 5
    atazanavir = 6
    lopinavir = 7
    darunavir = 8
    dolutegravir = 9
    cabotegravir = 10  # long acting PrEP

class ARTModule:
    vm_format = VmFormat.whb_lab  # TODO: check correct initial value? 
    vl_threshold = 1000
    time_of_first_vm = 0.5
    min_time_repeat_vm = 0.25  # 3 months?
    poc_vl_monitoring = False
    cd4_monitoring = False

    ## ART coverage changes
    lower_future_art_coverage = False


    rate_change_art_init_strategy = {
        ArtInitiationStrategy.cd4_lt_200_who4: 0.4,
        ArtInitiationStrategy.cd4_lt_350_pregnant: 0.4,
        ArtInitiationStrategy.cd4_lt_500_pregnant: 0.4,
        ArtInitiationStrategy.all_hiv_diagnosed: 0.4
    }

    def __init__(self):
        # set cd4_monitoring
        with importlib.resources.path("hivpy.data", "art.yaml") as data_path:
            self.art_data = ARTData(data_path)

        self.prob_base_init_ART = self.art_data.base_prob_init_ART.sample()
        self.prob_base_switch_line = self.art_data.base_prob_switch_line.sample()
        self.prob_vl_measurement_done = self.art_data.prob_vl_measurement_done.sample()

        # reduced interruption risk with point-of-care viral load monitoring
        self.reduced_interrupt_poc_vl = self.art_data.reduced_interrupt_poc_vl.sample()

        # additional "effective" adherence based on use of NNRTI due to long half-life
        self.effect_adherence_nnrti = 0.1 * np.exp(rng.normal(0.0, 0.3))

        self.base_rate_interruption = self.art_data.base_rate_interruption.sample()
        self.prob_lost_ART = self.art_data.prob_lost_ART.sample()
        self.base_rate_restart_ART = self.art_data.base_rate_restart_ART.sample()
        self.prob_supply_interrupted = self.art_data.prob_supply_interrupted
        self.prob_supply_resumed = self.art_data.prob_supply_resumed

        # rate that people are lost to follow up if average adherence is >=0.8
        self.base_rate_lost = self.art_data.base_rate_lost.sample()
        self.base_rate_return = self.art_data.base_rate_return.sample()
        self.base_rate_return_lencab = self.art_data.base_rate_return_lencab.sample()

        self.lower_future_art_coverage = self.art_data.lower_future_art_coverage.sample()
        self.higher_future_prep_oral_coverage = self.art_data.higher_future_prep_oral_coverage.sample()

    def init_strategies(self, pop: Population):
        pop.init_variable(col.HIV_MONITORING_STRATEGY, HivMonitoringStrategy.presence_tb_who4)
        pop.init_variable(col.ART_INITIATION_STRATEGY, ArtInitiationStrategy.all_tb_who4)
        pop.init_variable(col.ART_MONITORING_STRATEGY, ArtMonitoringStrategy.only_clinical)

    def update_strategies(self, current_date: date, pop: Population):
        """ Update strategies for HIV monitoring, ART initiation, and ART monitoring
            for all members of the population"""

        def apply_initiation_strategy(art_strategy: ArtInitiationStrategy,
                              start_date: date,
                              end_date: date,
                              hiv_strategy = None):
            if ((start_date <= current_date) and
                ((end_date is not None) and (current_date < end_date))
            ):
                r = rng.uniform(size=pop.size)
                new_strategy = pop.apply_bool_mask(r < self.rate_change_art_init_strategy[art_strategy])
                pop.set_present_variable(col.ART_INITIATION_STRATEGY, art_strategy, new_strategy)
                if(hiv_strategy is not None):
                    pop.set_present_variable(col.HIV_MONITORING_STRATEGY, hiv_strategy, new_strategy)

        apply_initiation_strategy(ArtInitiationStrategy.cd4_lt_200_who4,
                                date(2008, 1, 1),
                                date(2011, 6, 1),
                                HivMonitoringStrategy.cd4_6_monthly)
        
        apply_initiation_strategy(ArtInitiationStrategy.cd4_lt_350_pregnant,
                                date(2011, 6, 1),
                                date(2014, 1, 1))

        apply_initiation_strategy(ArtInitiationStrategy.cd4_lt_500_pregnant,
                                date(2014, 1, 1),
                                date(2016, 6, 1))
        
        apply_initiation_strategy(ArtInitiationStrategy.all_hiv_diagnosed,
                                date(2016, 6, 1),
                                None,
                                HivMonitoringStrategy.presence_tb_who4)
        
        if current_date >= date(2016, 3, 1):
            pop.set_present_variable(col.ART_MONITORING_STRATEGY, ArtMonitoringStrategy.vl_monitor_who)
            self.vm_format = VmFormat.whb_lab
            self.vl_threshold = 1000
            self.time_of_first_vm = 0.5
            self.min_time_repeat_vm = 0.25
            if(self.poc_vl_monitoring):
                self.vm_format = VmFormat.whb_poc

        if ((current_date >= date(2016, 6, 1)) and self.cd4_monitoring):
            pop.set_present_variable(col.ART_MONITORING_STRATEGY, ArtMonitoringStrategy.only_cd4_monitor)
            
        if (current_date >= date(2026, 1, 1)):
            people_on_cab_len = pop.get_sub_pop(OR(COND(col.ON_CAB, op.eq, True),
                                                   COND(col.ON_LEN, op.eq, True)))
            pop.set_present_variable(col.ART_MONITORING_STRATEGY, ArtMonitoringStrategy.on_len_cab, people_on_cab_len)
        
        # Changes in ART converage and oral PrEP coverage after year of intervention
        # only happens once
        # FIXME: what if the timestep doesn't divide the year exactly so we don't fulfil this equality?
        if(current_date == date(pop.policy_intervention_year, 1, 1)):
            if (self.lower_future_art_coverage):
                pop.scale_present_variable(col.RATE_CHOOSE_INTERRUPTION, 1.25)
                pop.scale_present_variable(col.PROB_LOSS_DIAGNOSIS, 1.25)
                pop.scale_present_variable(col.PROB_LOSS_ADC_TB, 1.25)
                pop.scale_present_variable(col.PROB_LOSS_WHO3, 1.25)
                pop.scale_present_variable(col.PROB_LOSS_ART, 1.25)
                pop.scale_present_variable(col.RATE_LOST, 1.25)

                pop.scale_present_variable(col.RATE_RESTART, 0.8)
                pop.scale_present_variable(col.RATE_RETURN, 0.8)
                pop.scale_present_variable(col.PROB_ART_INIT, 0.8)
                pop.scale_present_variable(col.PROB_RETURN_ADC, 0.8)
        
        # TODO: Do we need higher prep oral coverage? In SAS is it always false
        
    def update_regimens(self, current_date: date, pop: Population):
        if(date(2019, 6, 1) <= current_date <= date(2021, 1, 1)):
            pop.set_present_variable(col.ART_REGIMEN_OPT, 120)
    
        if(current_date >= date(2021, 1, 1)):
            pop.set_present_variable(col.ART_REGIMEN_OPT, 104)
        