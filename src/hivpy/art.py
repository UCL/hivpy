from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import importlib.resources
import operator as op
from enum import Enum

import numpy as np

import hivpy.column_names as col

from .art_data import ARTData
from .common import AND, COND, OR, Date, TimeDelta, is_in, past, rng
from .prep import PrEPType


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


class Interrupts(Enum):
    Not = 0
    Choice = 1
    Supply = 2
    Toxicity = 3


class ARTModule:
    vm_format = VmFormat.whb_lab  # TODO: check correct initial value?
    vl_threshold = 1000
    time_of_first_vm = 0.5
    min_time_repeat_vm = 0.25  # 3 months?
    poc_vl_monitoring = False
    cd4_monitoring = False

    # ART coverage changes
    lower_future_art_coverage = False

    rate_change_art_init_strategy = {
        ArtInitiationStrategy.cd4_lt_200_who4: 0.4,
        ArtInitiationStrategy.cd4_lt_350_pregnant: 0.4,
        ArtInitiationStrategy.cd4_lt_500_pregnant: 0.4,
        ArtInitiationStrategy.all_hiv_diagnosed: 0.4,
    }

    def __init__(self, pop: Population):
        # set cd4_monitoring
        with importlib.resources.path("hivpy.data", "art.yaml") as data_path:
            self.art_data = ARTData(data_path)

        self.pop = pop  # module has a reference to the population data

        self.art_intro_date = 2004
        self.prob_cd4_measure_done = 0.85
        self.sigma_measured_cd4 = 1.7

        self.base_prob_init_ART = self.art_data.base_prob_init_ART.sample()
        self.base_prob_switch_line = self.art_data.base_prob_switch_line.sample()
        self.prob_vl_measurement_done = self.art_data.prob_vl_measurement_done.sample()

        # reduced interruption risk with point-of-care viral load monitoring
        self.reduced_interrupt_poc_vl = self.art_data.reduced_interrupt_poc_vl.sample()

        # additional "effective" adherence based on use of NNRTI due to long half-life
        self.effect_adherence_nnrti = 0.1 * np.exp(rng.normal(0.0, 0.3))

        self.base_rate_interruption = self.art_data.base_rate_interruption.sample()
        self.base_prob_lost_ART = self.art_data.prob_lost_ART.sample()
        self.base_prob_loss_at_diagnosis = self.art_data.prob_loss_at_diagnosis.sample()
        self.base_prob_loss_adc_tb = self.art_data.prob_loss_adc_tb.sample()
        self.base_prob_loss_who3 = self.art_data.prob_loss_non_tb_who3.sample()
        self.base_rate_restart_ART = self.art_data.base_rate_restart_ART.sample()
        self.prob_supply_interrupted = self.art_data.prob_supply_interrupted
        self.prob_supply_resumed = self.art_data.prob_supply_resumed

        # rate that people are lost to follow up if average adherence is >=0.8
        self.base_rate_lost = self.art_data.base_rate_lost.sample()
        self.base_rate_return = self.art_data.base_rate_return.sample()
        self.base_rate_return_adc = self.art_data.rate_return_adc.sample()
        self.base_rate_return_lencab = self.art_data.base_rate_return_lencab.sample()

        self.lower_future_art_coverage = (
            self.art_data.lower_future_art_coverage.sample()
        )
        self.higher_future_prep_oral_coverage = (
            self.art_data.higher_future_prep_oral_coverage.sample()
        )

        # interrupt
        self.toxicity_interrupt_factor = (
            self.art_data.toxicity_interrupt_factor.sample()
        )
        self.prob_interrupt_choice = self.art_data.prob_interrupt_choice.sample()
        self.lencab_interrupt_factor = self.art_data.lencab_interrupt_factor.sample()
        self.vl_monitoring_interrupt_factor = (
            self.art_data.vl_monitoring_interrupt_factor.sample()
        )
        self.higher_newp_less_engagement = (
            self.art_data.higher_newp_less_engagement.sample()
        )
        self.higher_newp_interrupt_factor = 1.5
        self.prob_clinic_unaware_interrupt = (
            self.art_data.prob_clinic_unaware_interrupt.sample()
        )
        self.prob_supply_interrupt = 0.003

        self.sw_art_disadvantage = self.art_data.sw_art_disadvantage.sample()
        self.sw_interrupt_factor = (
            self.art_data.sw_interrupt_factor.sample()
            if self.sw_art_disadvantage
            else 1
        )
        self.sw_adherence_factor = (
            self.art_data.sw_adherence_factor.sample()
            if self.sw_art_disadvantage
            else 1
        )
        self.sw_loss_diagnosis_factor = (
            self.art_data.sw_loss_diagnosis_factor.sample()
            if self.sw_art_disadvantage
            else 1
        )

    def init_ART_columns(self):
        self.pop.init_variable(col.DATE_START_ART, None)
        self.pop.init_variable(col.CLINIC_VISIT, False)
        self.pop.init_variable(col.ART_NAIVE, True, 1)
        self.pop.init_variable(col.ON_ART, False, dt=1)
        self.pop.init_variable(col.ART_REGIMEN_OPT, 0)
        self.pop.init_variable(col.ABSENCE_CD4_YEAR_I, False)
        self.pop.init_variable(col.ABSENCE_CD4_YEAR_I, False)
        self.pop.init_variable(col.ART_START_DATE, None)
        self.init_strategies()
        self.pop.init_variable(col.FIRST_LINE_REGIMEN, 0)
        self.pop.init_variable(
            col.RATE_CHOOSE_INTERRUPTION, self.base_rate_interruption
        )
        self.pop.init_variable(
            col.PROB_LOSS_DIAGNOSIS, self.base_prob_loss_at_diagnosis
        )
        self.pop.init_variable(col.PROB_LOSS_ADC_TB, self.base_prob_loss_adc_tb)
        self.pop.init_variable(col.PROB_LOSS_WHO3, self.base_prob_loss_who3)
        self.pop.init_variable(col.PROB_LOSS_ART, self.base_prob_lost_ART)
        self.pop.init_variable(col.RATE_LOST, self.base_rate_lost)
        self.pop.init_variable(col.RATE_RESTART, self.base_rate_restart_ART)
        self.pop.init_variable(col.RATE_RETURN, self.base_rate_return)
        self.pop.init_variable(col.PROB_ART_INIT, self.base_prob_init_ART)
        self.pop.init_variable(col.PROB_RETURN_ADC, self.base_rate_return_adc)
        self.pop.init_variable(col.PROB_SWITCH_LINE, self.base_prob_switch_line)
        self.pop.init_variable(col.PROB_VL_MEASURE, self.prob_vl_measurement_done)
        self.pop.init_variable(col.CD4_MEASUREMENT, None, 2)
        self.pop.init_variable(col.ART_INTERRUPT, 0)
        self.pop.init_variable(col.ART_STOP_TOXICITY, False)
        self.pop.init_variable(col.ART_ADHERENCE, 0, dt=1)
        self.pop.init_variable(col.CURRENT_TOXICITY, False, dt=1)
        self.pop.init_variable(col.INJECTION_SITE_REACTION, False, dt=1)
        self.pop.init_variable(col.TIME_ON_ART, 0)
        self.pop.init_variable(col.SW_INTERRUPT_FACTOR, self.sw_interrupt_factor)
        self.pop.init_variable(col.CLINIC_UNAWARE_INTERRUPT, False)

    def init_strategies(self):
        self.pop.init_variable(
            col.HIV_MONITORING_STRATEGY, HivMonitoringStrategy.presence_tb_who4
        )
        self.pop.init_variable(
            col.ART_INITIATION_STRATEGY, ArtInitiationStrategy.all_tb_who4
        )
        self.pop.init_variable(
            col.ART_MONITORING_STRATEGY, ArtMonitoringStrategy.only_clinical
        )

    def update_strategies(self, current_date: Date):
        """Update strategies for HIV monitoring, ART initiation, and ART monitoring
        for all members of the population"""

        def apply_initiation_strategy(
            art_strategy: ArtInitiationStrategy,
            start_date: Date,
            end_date: Date,
            hiv_strategy=None,
        ):
            if (start_date <= current_date) and (
                (end_date is not None) and (current_date < end_date)
            ):
                r = rng.uniform(size=self.pop.size)
                new_strategy = self.pop.apply_bool_mask(
                    r < self.rate_change_art_init_strategy[art_strategy]
                )
                self.pop.set_present_variable(
                    col.ART_INITIATION_STRATEGY, art_strategy, new_strategy
                )
                if hiv_strategy is not None:
                    self.pop.set_present_variable(
                        col.HIV_MONITORING_STRATEGY, hiv_strategy, new_strategy
                    )

        apply_initiation_strategy(
            ArtInitiationStrategy.cd4_lt_200_who4,
            Date(2008, 1, 1),
            Date(2011, 6, 1),
            HivMonitoringStrategy.cd4_6_monthly,
        )

        apply_initiation_strategy(
            ArtInitiationStrategy.cd4_lt_350_pregnant,
            Date(2011, 6, 1),
            Date(2014, 1, 1),
        )

        apply_initiation_strategy(
            ArtInitiationStrategy.cd4_lt_500_pregnant,
            Date(2014, 1, 1),
            Date(2016, 6, 1),
        )

        apply_initiation_strategy(
            ArtInitiationStrategy.all_hiv_diagnosed,
            Date(2016, 6, 1),
            None,
            HivMonitoringStrategy.presence_tb_who4,
        )

        if current_date >= Date(2016, 3, 1):
            self.pop.set_present_variable(
                col.ART_MONITORING_STRATEGY, ArtMonitoringStrategy.vl_monitor_who
            )
            self.vm_format = VmFormat.whb_lab
            self.vl_threshold = 1000
            self.time_of_first_vm = 0.5
            self.min_time_repeat_vm = 0.25
            if self.poc_vl_monitoring:
                self.vm_format = VmFormat.whb_poc

        if (current_date >= Date(2016, 6, 1)) and self.cd4_monitoring:
            self.pop.set_present_variable(
                col.ART_MONITORING_STRATEGY, ArtMonitoringStrategy.only_cd4_monitor
            )

        if current_date >= Date(2026, 1, 1):
            people_on_cab_len = pop.get_sub_pop(
                OR(COND(col.ON_CAB, op.eq, True), COND(col.ON_LEN, op.eq, True))
            )
            self.pop.set_present_variable(
                col.ART_MONITORING_STRATEGY,
                ArtMonitoringStrategy.on_len_cab,
                people_on_cab_len,
            )

        # Changes in ART converage and oral PrEP coverage after year of intervention
        # only happens once
        # FIXME: what if the timestep doesn't divide the year exactly so we don't fulfil this equality?
        if current_date == Date(self.pop.policy_intervention_year, 1, 1):
            if self.lower_future_art_coverage:
                self.pop.scale_present_variable(col.RATE_CHOOSE_INTERRUPTION, 1.25)
                self.pop.scale_present_variable(col.PROB_LOSS_DIAGNOSIS, 1.25)
                self.pop.scale_present_variable(col.PROB_LOSS_ADC_TB, 1.25)
                self.pop.scale_present_variable(col.PROB_LOSS_WHO3, 1.25)
                self.pop.scale_present_variable(col.PROB_LOSS_ART, 1.25)
                self.pop.scale_present_variable(col.RATE_LOST, 1.25)

                self.pop.scale_present_variable(col.RATE_RESTART, 0.8)
                self.pop.scale_present_variable(col.RATE_RETURN, 0.8)
                self.pop.scale_present_variable(col.PROB_ART_INIT, 0.8)
                self.pop.scale_present_variable(col.PROB_RETURN_ADC, 0.8)

    def update_regimens(self, current_date: Date):
        if Date(2019, 6, 1) <= current_date <= Date(2021, 1, 1):
            self.pop.set_present_variable(col.ART_REGIMEN_OPT, 120)

        if current_date >= Date(2021, 1, 1):
            self.pop.set_present_variable(col.ART_REGIMEN_OPT, 125)

        flr_1 = self.pop.get_sub_pop(COND(col.ART_REGIMEN_OPT, op.eq, 107))
        self.pop.set_present_variable(col.FIRST_LINE_REGIMEN, 1, flr_1)

        flr_2 = self.pop.get_sub_pop(
            COND(
                col.ART_REGIMEN_OPT,
                is_in,
                [102, 103, 104, 105, 106, 113, 115, 116, 117, 118, 119, 120, 121, 125],
            )
        )
        self.pop.set_present_variable(col.FIRST_LINE_REGIMEN, 2, flr_2)

        flr_3 = self.pop.get_sub_pop(COND(col.ART_REGIMEN_OPT, op.eq, 130))
        self.pop.set_present_variable(col.FIRST_LINE_REGIMEN, 3, flr_3)

        reg_108 = self.pop.get_sub_pop(COND(col.ART_REGIMEN_OPT, op.eq, 108))
        self.pop.set_present_variable(col.PROB_SWITCH_LINE, 0.85, reg_108)
        self.pop.set_present_variable(col.PROB_VL_MEASURE, 0.85, reg_108)

        def set_absence_vl_strategy_by_regim(person):
            art_reg = person[col.ART_REGIMEN_OPT]
            art_start = person[col.ART_START_DATE]
            monitoring_strategy = 1  # default if nothing else modifies it
            if art_reg in [101, 102, 103, 104, 107, 110, 113, 116, 120, 121, 125, 130]:
                monitoring_strategy = 1500
            elif art_reg in [105, 106, 108, 109, 111, 112, 114]:
                monitoring_strategy = 153
            elif art_reg in [115, 117, 118, 119]:
                monitoring_strategy = 1500

            if art_reg in [112, 114] and ((current_date - art_start) > 1):
                monitoring_strategy = 150

            if current_date >= 2026 and person[col.ON_CAB] and person[col.ON_LEN]:
                monitoring_strategy = 1700

            person[col.ART_MONITORING_STRATEGY] = monitoring_strategy

        absence_vl_pop = self.pop.get_sub_pop(COND(col.ABSENCE_VL_YEAR_I, op.eq, True))
        self.pop.apply_function(
            set_absence_vl_strategy_by_regim, sub_pop=absence_vl_pop
        )

    def measure_CD4(self, current_date: Date):
        subpop = self.pop.get_sub_pop(
            AND(
                COND(col.HIV_MONITORING_STRATEGY, op.eq, 2),
                COND(col.HIV_STATUS, op.eq, True),
                COND(col.ART_NAIVE, op.eq, True),
                COND(col.CLINIC_VISIT, op.eq, True),
                COND(
                    col.DATE_LAST_CD4_MEASURE,
                    lambda t, dt: (t is None) or (current_date - t) > dt,
                    TimeDelta(months=3),
                ),
            )
        )
        measured = self.pop.apply_bool_mask(
            rng.uniform(size=len(subpop)) < self.prob_cd4_measure_done, subpop
        )
        n_measured = len(measured)
        cd4s = self.pop.get_variable(col.CD4, measured)
        cd4_measured = (
            np.sqrt(cd4s) + rng.normal(0, self.sigma_measured_cd4, size=n_measured)
        ) ** 2
        self.pop.set_present_variable(col.CD4_MEASUREMENT, cd4_measured, measured)
        self.pop.set_present_variable(col.DATE_LAST_CD4_MEASURE, current_date, measured)

    def initiate_ART(self, current_date: Date):
        hiv_pos_never_art = self.pop.get_sub_pop(
            AND(
                COND(col.HIV_STATUS, op.eq, True), COND(col.DATE_START_ART, op.eq, None)
            )
        )

        def init_art(person):
            art_init_strategy = person[col.ART_INITIATION_STRATEGY]
            hiv_monitoring_strategy = person[col.HIV_MONITORING_STRATEGY]

            recent_tb = (
                True
                if (person[col.TB_INFECTION_DATE] is not None)
                and (person.col[col.TB_INFECTION_DATE] < TimeDelta(months=6))
                else False
            )

            def probabilistically_set_ART_init(prob_factor=1):
                if rng.uniform() < (person[col.PROB_ART_INIT] * prob_factor):
                    person[col.DATE_START_ART] = current_date

            def check_cd4_measurements(limit):
                for dt in range(3):
                    cd4_column = self.pop.get_correct_column(col.CD4_MEASUREMENT, dt)
                    cd4_measurement = person[cd4_column]
                    if cd4_measurement is not None and cd4_measurement < limit:
                        return True
                return False

            if (
                current_date < self.art_intro_date
                and person[col.ART_NAIVE]
                and person[col.CLINIC_VISIT]
            ):

                if art_init_strategy == ArtInitiationStrategy.all_who4:
                    if person[col.EVER_WHO4]:
                        probabilistically_set_ART_init(0.5)

                elif art_init_strategy == ArtInitiationStrategy.all_tb_who4:
                    if person[col.EVER_WHO4] or recent_tb:
                        probabilistically_set_ART_init(0.5)

                elif art_init_strategy == ArtInitiationStrategy.all_hiv_diagnosed:
                    prob_factor = 0.5 if (person[col.EVER_WHO4] or recent_tb) else 1
                    if person[col.PREGNANT]:
                        prob_factor /= 4
                    probabilistically_set_ART_init(prob_factor)

                elif (art_init_strategy == 4) and (hiv_monitoring_strategy == 2):
                    if (
                        person[col.EVER_WHO4]
                        or recent_tb
                        or check_cd4_measurements(200)
                    ):
                        prob_factor = 0.5 if (person[col.EVER_WHO4] or recent_tb) else 1
                        probabilistically_set_ART_init(prob_factor)

                elif (art_init_strategy == 5) and (hiv_monitoring_strategy == 2):
                    if (
                        person[col.EVER_WHO4]
                        or recent_tb
                        or check_cd4_measurements(200)
                    ):
                        prob_factor = 0.5 if (person[col.EVER_WHO4] or recent_tb) else 1
                        probabilistically_set_ART_init(prob_factor)

                elif (art_init_strategy in [6, 9]) and (hiv_monitoring_strategy == 2):
                    if (
                        person[col.EVER_WHO4]
                        or recent_tb
                        or check_cd4_measurements(350)
                    ):
                        probabilistically_set_ART_init()

                elif (art_init_strategy in [3, 9, 10]) and person[col.PREGNANT]:
                    probabilistically_set_ART_init()

                elif (art_init_strategy == 10) and (hiv_monitoring_strategy == 2):
                    if (
                        person[col.EVER_WHO4]
                        or recent_tb
                        or check_cd4_measurements(500)
                    ):
                        probabilistically_set_ART_init()

        self.pop.apply_function(init_art, sub_pop=hiv_pos_never_art)

    def ART_interruption(self):
        # reset any interruption data
        self.pop.set_present_variable(col.ART_INTERRUPT, Interrupts.Not)

        # Interruption due to "choice" as opposed to drug toxicity
        not_toxicity = self.pop.get_sub_pop(
            AND(
                COND(col.HIV_STATUS, op.eq, True),
                COND(col.ART_STOP_TOXICITY, op.eq, False),
                COND(past(col.ON_ART, dt=1), op.eq, True),
            )
        )

        def stop_by_choice(person):
            prev_adherence = person[past(col.ART_ADHERENCE, dt=1)]
            recent_len = person[col.PREP_TYPE] == PrEPType.Lenacapavir and person[
                col.LAST_PREP_USE_DATE
            ] > (self.pop.date - TimeDelta(months=5))
            prev_toxicity = person[past(col.CURRENT_TOXICITY, dt=1)]

            prob_interrupt = self.prob_interrupt_choice
            if not recent_len:
                if 0.5 <= prev_adherence < 0.8:
                    prob_interrupt *= 1.5
                elif prev_adherence < 0.5:
                    prob_interrupt *= 2

            if prev_toxicity:
                prob_interrupt *= self.toxicity_interrupt_factor

            if person[col.ON_LEN]:
                prob_interrupt *= self.lencab_interrupt_factor
                if person[col.INJECTION_SITE_REACTION]:
                    prob_interrupt *= 1.1

            if person[col.PREGNANT]:
                prob_interrupt *= 0.01

            if person[col.TIME_ON_ART] > 0.25:
                prob_interrupt *= 0.5

            if person[col.SEX_WORKER]:
                prob_interrupt = min(
                    1, prob_interrupt * person[col.SW_INTERRUPT_FACTOR]
                )

            if person[col.ART_MONITORING_STRATEGY] == 150 and self.vm_format in [3, 4]:
                prob_interrupt *= self.vl_monitoring_interrupt_factor

            if self.higher_newp_less_engagement:
                prob_interrupt *= self.higher_newp_interrupt_factor

            if rng.uniform() < prob_interrupt:
                person[col.ART_INTERRUPT] = Interrupts.Choice
                person[col.CLINIC_UNAWARE_INTERRUPT] = (
                    rng.uniform() < self.prob_clinic_unaware_interrupt
                )

        self.pop.apply_function(stop_by_choice, not_toxicity)

        # interruption due to interruption of drug supply
        art_clinic_visitors = self.pop.get_sub_pop_intersection(
            not_toxicity,
            self.pop.get_sub_pop(
                AND(
                    COND(col.CLINIC_VISIT, op.eq, True),
                    COND(col.ART_INTERRUPT, op.eq, 0),
                )
            ),
        )
        n_visitors = len(art_clinic_visitors)
        supply_interruptions = rng.uniform(size=n_visitors) < self.prob_supply_interrupt
        self.pop.set_present_variable(
            col.ART_INTERRUPT,
            Interrupts.Supply,
            self.pop.apply_bool_mask(supply_interruptions, art_clinic_visitors),
        )

