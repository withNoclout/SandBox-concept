'use client';

import React, { useState, useEffect } from 'react';
import { Icon } from '@/components/ui/Icon';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ComposedChart, Area, BarChart, Bar, Cell } from 'recharts';

interface BatteryStats {
    cycleCount: number;
    maxCapacity: number;
    designCapacity: number;
    temperature: number;
    voltage: number;
    isCharging: boolean;
    soh: number;
}

interface SimulationPoint {
    cycle: number;
    soh: number;
}



export default function SOHSimulation() {
    // State
    const [stats, setStats] = useState<BatteryStats | null>(null);
    const [history, setHistory] = useState<SimulationPoint[]>([]);
    const [isLoading, setIsLoading] = useState(false);

    // Mock Data (Memoized)
    const cccvData = React.useMemo(() => {
        const data = [];
        // CC Phase: 0 to 60 mins
        for (let t = 0; t <= 60; t++) {
            data.push({
                time: t,
                voltage: 3.0 + (1.2 * (t / 60)), // 3.0V -> 4.2V
                current: 1.0 // Constant 1.0C
            });
        }
        // CV Phase: 60 to 120 mins
        for (let t = 61; t <= 120; t++) {
            const progress = (t - 60) / 60;
            data.push({
                time: t,
                voltage: 4.2, // Constant 4.2V
                current: 1.0 * Math.exp(-4 * progress) // Decay
            });
        }
        return data;
    }, []);

    // Mock Thermal Stress Data (Memoized)
    const thermalData = React.useMemo(() => {
        const data = [];
        const T_REF = 25;
        let integralHigh = 0;
        let integralLow = 0;

        // Simulate 60 minutes of charging (20% -> 80%)
        for (let t = 0; t <= 60; t++) {
            // Scenario 1: High Temp (Starts at 30°C, rises to 45°C)
            const tempHigh = 30 + (15 * (t / 60));

            // Scenario 2: Low Temp (Starts at 10°C, rises to 20°C)
            const tempLow = 10 + (10 * (t / 60));

            // Integrate (T - Tref) * dt (assuming dt = 1 min)
            integralHigh += (tempHigh - T_REF);
            integralLow += (tempLow - T_REF);

            data.push({
                time: t,
                stressHigh: integralHigh,
                stressLow: integralLow
            });
        }
        return data;
    }, []);

    // Mock ICA Data (Memoized)
    const icaData = React.useMemo(() => {
        const data = [];
        // Simulate Voltage from 3.0V to 4.2V
        for (let v = 3.0; v <= 4.2; v += 0.01) {
            // Gaussian Peak Function
            const gaussian = (x: number, amp: number, mean: number, std: number) => {
                return amp * Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
            };

            // New Battery: Sharp peaks at 3.7V and 4.0V
            const newPeak1 = gaussian(v, 3.0, 3.7, 0.05);
            const newPeak2 = gaussian(v, 2.5, 4.0, 0.05);
            const newdQdV = newPeak1 + newPeak2 + 0.2; // Baseline offset

            // Old Battery: Lower, broader, shifted peaks (Resistance + Loss of Active Material)
            const oldPeak1 = gaussian(v, 2.0, 3.75, 0.08); // Shifted right, lower amp, wider std
            const oldPeak2 = gaussian(v, 1.5, 4.05, 0.08);
            const olddQdV = oldPeak1 + oldPeak2 + 0.2;

            data.push({
                voltage: Number(v.toFixed(2)),
                newBattery: newdQdV,
                oldBattery: olddQdV
            });
        }
        return data;
    }, []);

    // Mock Arrhenius Data (Memoized)
    const arrheniusData = React.useMemo(() => {
        const data = [];
        // Temp range: 10°C to 60°C
        // T(K) = T(C) + 273.15
        // k = A * exp(-Ea / RT)
        // ln(k) = ln(A) - (Ea/R) * (1/T)

        const Ea = 0.4; // eV
        const Kb = 8.617e-5; // eV/K (Boltzmann constant)
        // Ea/Kb approx 4640 K

        for (let tempC = 10; tempC <= 60; tempC += 2) {
            const tempK = tempC + 273.15;
            const invTemp = 1000 / tempK; // x-axis

            // Degradation Rate (Arbitrary units, normalized)
            // k ~ exp(-Ea / (Kb * T))
            const exponent = -Ea / (Kb * tempK);
            const k = Math.exp(exponent) * 1e6; // Scale for visibility
            const lnk = Math.log(k);

            data.push({
                tempC,
                invTemp: Number(invTemp.toFixed(3)),
                k: Number(k.toFixed(3)),
                lnk: Number(lnk.toFixed(3))
            });
        }
        return data;
    }, []);

    // Mock Maintenance Data (Memoized)
    const maintenanceData = React.useMemo(() => {
        const data = [];
        let sohUnmanaged = 100;
        let sohManaged = 100;

        // Simulate 1000 Cycles
        for (let cycle = 0; cycle <= 1000; cycle += 10) {
            // Unmanaged: Moderate-High Decay (EOL ~800 cycles)
            // Factors: High Temp (45C), 100% DOD
            // Real-world data suggests ~80% SOH after 500-800 cycles for heavy use
            let decayUnmanaged = 0.25;

            // Slight acceleration after 500 cycles
            if (cycle > 500) decayUnmanaged += 0.05;

            sohUnmanaged -= decayUnmanaged;

            // Managed: Slow Decay (EOL > 1200 cycles)
            // Factors: Controlled Temp (25C), 80% Charge Limit
            // Optimized Li-ion can last 1000+ cycles with >90% capacity
            const decayManaged = 0.1;
            sohManaged -= decayManaged;

            data.push({
                cycle,
                unmanaged: Math.max(0, Number(sohUnmanaged.toFixed(1))),
                managed: Math.max(0, Number(sohManaged.toFixed(1)))
            });
        }
        return data;
    }, []);

    // Mock AI Training Data (Memoized)
    const aiTrainingData = React.useMemo(() => {
        const data = [];
        for (let epoch = 1; epoch <= 50; epoch++) {
            // RMSE: Exponential decay
            const rmse = 0.4 * Math.exp(-0.1 * epoch) + 0.01 + (Math.random() * 0.005);

            // MAE: Slightly lower than RMSE
            const mae = rmse * 0.8 + (Math.random() * 0.002);

            // R2: Inverse exponential growth
            // R2 = 1 - (MSE / Var), simplified simulation
            const r2 = 0.99 * (1 - Math.exp(-0.15 * epoch)) - (Math.random() * 0.005);

            data.push({
                epoch,
                rmse: Number(rmse.toFixed(4)),
                mae: Number(mae.toFixed(4)),
                r2: Number(Math.max(0, r2).toFixed(4))
            });
        }
        return data;
    }, []);

    // Mock Batch Analysis Data (Memoized)
    const batchAnalysisData = React.useMemo(() => {
        // Model Definitions with Colors
        const models = [
            { name: '1D-CNN', color: '#ef4444' }, // Red
            { name: 'LSTM', color: '#f97316' }, // Orange
            { name: 'ConvLSTM', color: '#f59e0b' }, // Amber
            { name: 'Informer', color: '#eab308' }, // Yellow
            { name: 'BiLSTM', color: '#84cc16' }, // Lime
            { name: 'MBLSTM', color: '#06b6d4' }, // Cyan
            { name: 'iTransformer', color: '#3b82f6' }, // Blue
            { name: 'MBLSTM+Informer', color: '#8b5cf6' }, // Violet
            { name: 'MBLSTM+iTransformer', color: '#10b981', isBest: true }, // Emerald (Best)
        ];

        // RMSE Data from User Table
        const batchRmseData: Record<string, Record<string, number>> = {
            'Batch 1.1': {
                '1D-CNN': 0.0414, 'LSTM': 0.0245, 'ConvLSTM': 0.0124, 'Informer': 0.0289,
                'BiLSTM': 0.0064, 'MBLSTM': 0.0286, 'iTransformer': 0.0056, 'MBLSTM+Informer': 0.0286,
                'MBLSTM+iTransformer': 0.0037
            },
            'Batch 3.1': {
                '1D-CNN': 0.0217, 'LSTM': 0.0307, 'ConvLSTM': 0.0254, 'Informer': 0.0447,
                'BiLSTM': 0.0217, 'MBLSTM': 0.0444, 'iTransformer': 0.0078, 'MBLSTM+Informer': 0.0204,
                'MBLSTM+iTransformer': 0.0055
            },
            'Batch 4.1': {
                '1D-CNN': 0.0288, 'LSTM': 0.0483, 'ConvLSTM': 0.0272, 'Informer': 0.0407,
                'BiLSTM': 0.0163, 'MBLSTM': 0.0376, 'iTransformer': 0.0032, 'MBLSTM+Informer': 0.0387,
                'MBLSTM+iTransformer': 0.0031
            },
            'Batch 5.1': {
                '1D-CNN': 0.0174, 'LSTM': 0.0387, 'ConvLSTM': 0.0426, 'Informer': 0.0402,
                'BiLSTM': 0.0224, 'MBLSTM': 0.0382, 'iTransformer': 0.0118, 'MBLSTM+Informer': 0.0138,
                'MBLSTM+iTransformer': 0.0112
            }
        };

        const generateBatch = (batchId: string) => {
            const data = [];
            const totalCycles = 100; // Normalized cycle count for visualization
            const decayRate = 0.0015 + Math.random() * 0.001; // Random decay slope
            const rmseValues = batchRmseData[batchId];

            for (let cycle = 0; cycle <= totalCycles; cycle++) {
                // Ground Truth Generation based on Batch Pattern
                let actual = 1.0;

                if (batchId === 'Batch 1.1') {
                    // Steady Decrease (Linear/Slightly Exponential)
                    actual = 1.0 - (cycle * 0.002);
                } else if (batchId === 'Batch 3.1') {
                    // Wavy (Sinusoidal fluctuations)
                    // Decays but has a wave pattern
                    actual = 1.0 - (cycle * 0.0015) + (Math.sin(cycle * 0.5) * 0.015);
                } else if (batchId === 'Batch 4.1') {
                    // Nonlinear (Accelerating degradation)
                    // Starts slow, drops fast
                    actual = 1.0 - (Math.pow(cycle, 2) * 0.00004);
                } else if (batchId === 'Batch 5.1') {
                    // Irregular and Noisy
                    // Random walk + spikes
                    const trend = 1.0 - (cycle * 0.002);
                    const randomWalk = (Math.sin(cycle * 0.2) * 0.01) + ((Math.random() - 0.5) * 0.03);
                    actual = trend + randomWalk;
                }

                // Clamp actual to realistic bounds
                actual = Math.min(1.0, Math.max(0.5, actual));

                const point: any = { cycle, actual: Number(actual.toFixed(4)) };

                // Generate prediction for each model
                models.forEach(model => {
                    const rmse = rmseValues[model.name] || 0.01;

                    // Noise Magnitude proportional to RMSE
                    // We add a unique bias per model to separate the lines visually
                    const uniqueBias = Math.sin(cycle * 0.1 + model.name.length) * (rmse * 0.8);
                    const randomNoise = (Math.random() - 0.5) * (rmse * 2.0);

                    let predicted = actual + uniqueBias + randomNoise;

                    // Ensure it doesn't go above 1.0 or below 0
                    predicted = Math.min(1.0, Math.max(0, predicted));

                    point[model.name] = Number(predicted.toFixed(4));
                });

                data.push(point);
            }
            return { id: batchId, data, models };
        };

        return [
            generateBatch('Batch 1.1'),
            generateBatch('Batch 3.1'),
            generateBatch('Batch 4.1'),
            generateBatch('Batch 5.1'),
        ];
    }, []);

    // Mock LIME Explanation Data (Multi-Battery)
    const limeData = React.useMemo(() => [
        {
            id: 'Battery 2C_1',
            data: [
                { feature: 'voltage mean_T2 (<= 0.56)', value: -0.034, type: 'Negative' },
                { feature: 'voltage mean_T5 (<= 0.56)', value: -0.032, type: 'Negative' },
                { feature: 'voltage mean_T0 (<= 0.56)', value: -0.031, type: 'Negative' },
                { feature: 'voltage mean_T3 (<= 0.56)', value: -0.031, type: 'Negative' },
                { feature: 'voltage mean_T7 (<= 0.55)', value: -0.030, type: 'Negative' },
                { feature: 'current entropy_T4 (> 0.15)', value: -0.004, type: 'Negative' },
                { feature: 'voltage std_T0 (> 0.38)', value: -0.003, type: 'Negative' },
            ]
        },
        {
            id: 'Battery R2',
            data: [
                { feature: 'voltage mean_T4 (<= 0.51)', value: -0.038, type: 'Negative' },
                { feature: 'voltage mean_T2 (<= 0.51)', value: -0.036, type: 'Negative' },
                { feature: 'voltage mean_T3 (<= 0.51)', value: -0.036, type: 'Negative' },
                { feature: 'voltage mean_T6 (<= 0.50)', value: -0.035, type: 'Negative' },
                { feature: 'voltage mean_T0 (<= 0.51)', value: -0.034, type: 'Negative' },
                { feature: 'current std_T3 (<= 0.48)', value: 0.004, type: 'Positive' },
            ]
        },
        {
            id: 'Battery R_1',
            data: [
                { feature: 'voltage mean_T5 (<= 0.46)', value: -0.052, type: 'Negative' },
                { feature: 'voltage mean_T4 (<= 0.46)', value: -0.052, type: 'Negative' },
                { feature: 'voltage mean_T3 (<= 0.47)', value: -0.051, type: 'Negative' },
                { feature: 'voltage mean_T2 (<= 0.47)', value: -0.051, type: 'Negative' },
                { feature: 'voltage mean_T7 (<= 0.45)', value: -0.050, type: 'Negative' },
                { feature: 'voltage slope_T0 (> 0.48)', value: 0.005, type: 'Positive' },
            ]
        },
        {
            id: 'Battery RW_1',
            data: [
                { feature: 'voltage mean_T5 (<= 0.54)', value: -0.028, type: 'Negative' },
                { feature: 'voltage mean_T2 (<= 0.54)', value: -0.028, type: 'Negative' },
                { feature: 'voltage mean_T4 (<= 0.54)', value: -0.028, type: 'Negative' },
                { feature: 'voltage mean_T3 (<= 0.54)', value: -0.028, type: 'Negative' },
                { feature: 'voltage mean_T7 (<= 0.54)', value: -0.027, type: 'Negative' },
                { feature: 'CC charge time_T3 (> 0.02)', value: -0.004, type: 'Negative' },
            ]
        }
    ], []);



    // Parameters
    const [avgTemp, setAvgTemp] = useState(25);
    const [chargeLimit, setChargeLimit] = useState(80);
    const [dod, setDod] = useState(0.5);
    const [cRate, setCRate] = useState(0.5);

    // Fetch Real Data
    useEffect(() => {
        const fetchStats = async () => {
            try {
                const res = await fetch('/api/battery-stats');
                const data = await res.json();
                if (!data.error) {
                    setStats(data);
                    setAvgTemp(Math.round(data.temperature) || 25);
                }
            } catch (err) {
                console.error("Failed to fetch stats", err);
            }
        };
        fetchStats();
    }, []);

    // Run Simulation
    const runSimulation = async () => {
        setIsLoading(true);
        try {
            const res = await fetch('/api/soh-simulation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    avgTemp,
                    chargeLimit,
                    dod,
                    cRate,
                    initialSoh: stats?.soh || 100,
                    initialCycles: stats?.cycleCount || 0
                })
            });
            const data = await res.json();
            setHistory(data);
        } catch (err) {
            console.error("Simulation failed", err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="relative z-10 max-w-6xl mx-auto px-6 py-12 md:py-20 flex flex-col gap-12">

            {/* Header */}
            <header className="flex flex-col gap-4 animate-drift text-center items-center">
                <div className="mb-2">
                    <Icon icon="lucide:battery-charging" className="text-[#8f9196] w-10 h-10 md:w-12 md:h-12 animate-breathe" />
                </div>
                <h1 className="md:text-5xl lg:text-6xl leading-none text-3xl font-medium text-[#d9d7c5] tracking-tighter uppercase">
                    SOH_SIMULATOR // V.01
                </h1>
                <p className="max-w-xl mx-auto text-[#8f9196] text-xs md:text-sm leading-loose tracking-wide">
                    &gt; ANALYZING DEGRADATION VECTORS...<br />
                    Projecting entropy based on real-time telemetry.
                </p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* Left Column: Real Data & Controls */}
                <div className="flex flex-col gap-8">

                    {/* Real Data Card */}
                    <div className="group relative p-6 flex flex-col gap-4 hover-sketch cursor-default">
                        <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none"></div>
                        <div className="flex justify-between items-start">
                            <Icon icon="lucide:activity" className="text-[#d9d7c5] w-5 h-5" />
                            <span className="text-xs text-[#8f9196] font-bold">TELEMETRY</span>
                        </div>

                        {stats ? (
                            <div className="grid grid-cols-2 gap-4 mt-2">
                                <div>
                                    <p className="text-[10px] text-[#8f9196] uppercase tracking-widest">SOH</p>
                                    <p className="text-xl text-[#d9d7c5] font-bold">{stats.soh.toFixed(1)}%</p>
                                </div>
                                <div>
                                    <p className="text-[10px] text-[#8f9196] uppercase tracking-widest">CYCLES</p>
                                    <p className="text-xl text-[#d9d7c5] font-bold">{stats.cycleCount}</p>
                                </div>
                                <div>
                                    <p className="text-[10px] text-[#8f9196] uppercase tracking-widest">TEMP</p>
                                    <p className="text-xl text-[#d9d7c5] font-bold">{stats.temperature.toFixed(1)}°C</p>
                                </div>
                                <div>
                                    <p className="text-[10px] text-[#8f9196] uppercase tracking-widest">STATUS</p>
                                    <p className="text-xs text-[#d9d7c5] mt-1">{stats.isCharging ? 'CHARGING' : 'DISCHARGING'}</p>
                                </div>
                            </div>
                        ) : (
                            <p className="text-xs text-[#8f9196] animate-pulse">Scanning hardware...</p>
                        )}
                    </div>

                    {/* Controls */}
                    <div className="group relative p-6 flex flex-col gap-6 hover-sketch">
                        <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none" style={{ borderRadius: '15px 225px 15px 255px / 255px 15px 225px 15px' }}></div>
                        <div className="flex justify-between items-start">
                            <Icon icon="lucide:sliders" className="text-[#d9d7c5] w-5 h-5" />
                            <span className="text-xs text-[#8f9196] font-bold">PARAMETERS</span>
                        </div>

                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between text-xs text-[#8f9196] mb-1">
                                    <span>AVG TEMP</span>
                                    <span className="text-[#d9d7c5]">{avgTemp}°C</span>
                                </div>
                                <input type="range" min="10" max="60" value={avgTemp} onChange={(e) => setAvgTemp(Number(e.target.value))} className="w-full accent-[#d9d7c5] h-1 bg-[#8f9196]/30 rounded-lg appearance-none cursor-pointer" />
                            </div>

                            <div>
                                <div className="flex justify-between text-xs text-[#8f9196] mb-1">
                                    <span>CHARGE LIMIT</span>
                                    <span className="text-[#d9d7c5]">{chargeLimit}%</span>
                                </div>
                                <input type="range" min="50" max="100" value={chargeLimit} onChange={(e) => setChargeLimit(Number(e.target.value))} className="w-full accent-[#d9d7c5] h-1 bg-[#8f9196]/30 rounded-lg appearance-none cursor-pointer" />
                            </div>

                            <div>
                                <div className="flex justify-between text-xs text-[#8f9196] mb-1">
                                    <span>DEPTH OF DISCHARGE</span>
                                    <span className="text-[#d9d7c5]">{dod}</span>
                                </div>
                                <input type="range" min="0.1" max="1.0" step="0.1" value={dod} onChange={(e) => setDod(Number(e.target.value))} className="w-full accent-[#d9d7c5] h-1 bg-[#8f9196]/30 rounded-lg appearance-none cursor-pointer" />
                            </div>

                            <div>
                                <div className="flex justify-between text-xs text-[#8f9196] mb-1">
                                    <span>C-RATE</span>
                                    <span className="text-[#d9d7c5]">{cRate}C</span>
                                </div>
                                <input type="range" min="0.1" max="2.0" step="0.1" value={cRate} onChange={(e) => setCRate(Number(e.target.value))} className="w-full accent-[#d9d7c5] h-1 bg-[#8f9196]/30 rounded-lg appearance-none cursor-pointer" />
                            </div>
                        </div>

                        <button
                            onClick={runSimulation}
                            disabled={isLoading}
                            className="mt-2 w-full py-3 text-[#d9d7c5] border border-[#8f9196] hover:bg-[#d9d7c5]/10 transition-colors uppercase text-xs tracking-widest font-bold"
                        >
                            {isLoading ? 'CALCULATING...' : 'RUN_SIMULATION'}
                        </button>
                    </div>
                </div>

                {/* Right Column: Chart */}
                <div className="lg:col-span-2 group relative p-6 flex flex-col gap-4 hover-sketch min-h-[400px]">
                    <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none" style={{ borderRadius: '255px 15px 225px 15px / 15px 225px 15px 255px' }}></div>
                    <div className="flex justify-between items-start mb-4">
                        <Icon icon="lucide:line-chart" className="text-[#d9d7c5] w-5 h-5" />
                        <span className="text-xs text-[#8f9196] font-bold">PROJECTION_PLOT</span>
                    </div>

                    <div className="flex-1 w-full h-full min-h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={history}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#8f9196" opacity={0.1} />
                                <XAxis
                                    dataKey="cycle"
                                    stroke="#8f9196"
                                    tick={{ fontSize: 10 }}
                                    label={{ value: 'CYCLES', position: 'insideBottomRight', offset: -5, fill: '#8f9196', fontSize: 10 }}
                                />
                                <YAxis
                                    domain={[60, 100]}
                                    stroke="#8f9196"
                                    tick={{ fontSize: 10 }}
                                    label={{ value: 'SOH %', angle: -90, position: 'insideLeft', fill: '#8f9196', fontSize: 10 }}
                                />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#8f9196', color: '#d9d7c5' }}
                                    itemStyle={{ color: '#d9d7c5' }}
                                />
                                <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'EOL (80%)', fill: '#ef4444', fontSize: 10 }} />
                                <Line
                                    type="monotone"
                                    dataKey="soh"
                                    stroke="#d9d7c5"
                                    strokeWidth={2}
                                    dot={false}
                                    activeDot={{ r: 4, fill: '#d9d7c5' }}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* CC-CV Charging Profile (Mock) */}
            <div className="group relative p-6 flex flex-col gap-4 hover-sketch min-h-[400px]">
                <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none" style={{ borderRadius: '20px 225px 20px 230px / 230px 20px 225px 20px' }}></div>
                <div className="flex justify-between items-start mb-4">
                    <Icon icon="lucide:zap" className="text-[#d9d7c5] w-5 h-5" />
                    <span className="text-xs text-[#8f9196] font-bold">CHARGING_PROFILE (CC-CV MOCK)</span>
                </div>

                <div className="w-full h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={cccvData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#8f9196" opacity={0.1} />
                            <XAxis
                                dataKey="time"
                                stroke="#8f9196"
                                tick={{ fontSize: 10 }}
                                label={{ value: 'TIME (MIN)', position: 'insideBottomRight', offset: -5, fill: '#8f9196', fontSize: 10 }}
                            />
                            <YAxis
                                yAxisId="left"
                                domain={[2.5, 4.5]}
                                stroke="#d9d7c5"
                                tick={{ fontSize: 10 }}
                                label={{ value: 'VOLTAGE (V)', angle: -90, position: 'insideLeft', fill: '#d9d7c5', fontSize: 10 }}
                            />
                            <YAxis
                                yAxisId="right"
                                orientation="right"
                                domain={[0, 1.2]}
                                stroke="#8f9196"
                                tick={{ fontSize: 10 }}
                                label={{ value: 'CURRENT (C)', angle: 90, position: 'insideRight', fill: '#8f9196', fontSize: 10 }}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#8f9196', color: '#d9d7c5' }}
                                itemStyle={{ color: '#d9d7c5' }}
                            />
                            <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="voltage"
                                stroke="#d9d7c5"
                                strokeWidth={2}
                                dot={false}
                                name="Voltage"
                            />
                            <Area
                                yAxisId="right"
                                type="monotone"
                                dataKey="current"
                                fill="#8f9196"
                                stroke="#8f9196"
                                fillOpacity={0.1}
                                name="Current"
                            />
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Thermal Stress Chart (Mock) */}
            <div className="group relative p-6 flex flex-col gap-4 hover-sketch min-h-[500px]">
                <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none" style={{ borderRadius: '15px 255px 15px 225px / 225px 15px 255px 15px' }}></div>
                <div className="flex justify-between items-start mb-4">
                    <Icon icon="lucide:thermometer" className="text-[#d9d7c5] w-5 h-5" />
                    <span className="text-xs text-[#8f9196] font-bold">THERMAL_STRESS_INTEGRAL (∫(T-Tref)dt)</span>
                </div>

                <div className="w-full h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={thermalData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#8f9196" opacity={0.1} />
                            <XAxis
                                dataKey="time"
                                stroke="#8f9196"
                                tick={{ fontSize: 10 }}
                                label={{ value: 'TIME (MIN)', position: 'insideBottomRight', offset: -5, fill: '#8f9196', fontSize: 10 }}
                            />
                            <YAxis
                                stroke="#8f9196"
                                tick={{ fontSize: 10 }}
                                label={{ value: 'ACCUMULATED STRESS (°C·min)', angle: -90, position: 'insideLeft', fill: '#8f9196', fontSize: 10 }}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#8f9196', color: '#d9d7c5' }}
                                itemStyle={{ color: '#d9d7c5' }}
                            />
                            <Line
                                type="monotone"
                                dataKey="stressHigh"
                                stroke="#ef4444"
                                strokeWidth={2}
                                dot={false}
                                name="High Temp Scenario"
                            />
                            <Line
                                type="monotone"
                                dataKey="stressLow"
                                stroke="#3b82f6"
                                strokeWidth={2}
                                dot={false}
                                name="Low Temp Scenario"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* ICA (dQ/dV) Analysis (Mock) */}
            <div className="group relative p-6 flex flex-col gap-4 hover-sketch min-h-[500px]">
                <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none" style={{ borderRadius: '255px 15px 225px 15px / 15px 225px 15px 255px' }}></div>
                <div className="flex justify-between items-start mb-4">
                    <Icon icon="lucide:microscope" className="text-[#d9d7c5] w-5 h-5" />
                    <span className="text-xs text-[#8f9196] font-bold">INCREMENTAL_CAPACITY_ANALYSIS (dQ/dV)</span>
                </div>

                <div className="w-full h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={icaData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#8f9196" opacity={0.1} />
                            <XAxis
                                dataKey="voltage"
                                stroke="#8f9196"
                                tick={{ fontSize: 10 }}
                                domain={[3.0, 4.2]}
                                type="number"
                                label={{ value: 'VOLTAGE (V)', position: 'insideBottomRight', offset: -5, fill: '#8f9196', fontSize: 10 }}
                            />
                            <YAxis
                                stroke="#8f9196"
                                tick={{ fontSize: 10 }}
                                label={{ value: 'dQ/dV (Ah/V)', angle: -90, position: 'insideLeft', fill: '#8f9196', fontSize: 10 }}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#8f9196', color: '#d9d7c5' }}
                                itemStyle={{ color: '#d9d7c5' }}
                            />
                            <Line
                                type="monotone"
                                dataKey="newBattery"
                                stroke="#10b981"
                                strokeWidth={2}
                                dot={false}
                                name="New Battery (100% SOH)"
                            />
                            <Line
                                type="monotone"
                                dataKey="oldBattery"
                                stroke="#f59e0b"
                                strokeWidth={2}
                                dot={false}
                                name="Aged Battery (80% SOH)"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Arrhenius Plot (Mock) */}
            <div className="group relative p-6 flex flex-col gap-4 hover-sketch min-h-[500px]">
                <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none" style={{ borderRadius: '20px 225px 20px 230px / 230px 20px 225px 20px' }}></div>
                <div className="flex justify-between items-start mb-4">
                    <Icon icon="lucide:flame" className="text-[#d9d7c5] w-5 h-5" />
                    <span className="text-xs text-[#8f9196] font-bold">ARRHENIUS_PLOT (TEMP vs DEGRADATION)</span>
                </div>

                <div className="w-full h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={arrheniusData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#8f9196" opacity={0.1} />
                            <XAxis
                                dataKey="tempC"
                                stroke="#8f9196"
                                tick={{ fontSize: 10 }}
                                type="number"
                                domain={['auto', 'auto']}
                                label={{ value: 'TEMPERATURE (°C)', position: 'insideBottomRight', offset: -5, fill: '#8f9196', fontSize: 10 }}
                            />
                            <YAxis
                                stroke="#8f9196"
                                tick={{ fontSize: 10 }}
                                label={{ value: 'DEGRADATION RATE (k)', angle: -90, position: 'insideLeft', fill: '#8f9196', fontSize: 10 }}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#8f9196', color: '#d9d7c5' }}
                                itemStyle={{ color: '#d9d7c5' }}
                                labelFormatter={(value) => `Temp: ${value}°C`}
                                formatter={(value: number) => [value.toFixed(4), 'Rate (k)']}
                            />
                            <Line
                                type="monotone"
                                dataKey="k"
                                stroke="#ef4444"
                                strokeWidth={2}
                                dot={{ r: 4, fill: '#ef4444' }}
                                name="Degradation Rate"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>


            {/* Maintenance Comparison (Unmanaged vs Managed) */}
            <div className="group relative p-6 flex flex-col gap-4 hover-sketch min-h-[500px]">
                <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none" style={{ borderRadius: '15px 255px 15px 225px / 225px 15px 255px 15px' }}></div>
                <div className="flex justify-between items-start mb-4">
                    <Icon icon="lucide:shield-check" className="text-[#d9d7c5] w-5 h-5" />
                    <span className="text-xs text-[#8f9196] font-bold">MAINTENANCE_SCENARIOS (PREDICTIVE vs REACTIVE)</span>
                </div>

                <div className="w-full h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={maintenanceData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#8f9196" opacity={0.1} />
                            <XAxis
                                dataKey="cycle"
                                stroke="#8f9196"
                                tick={{ fontSize: 10 }}
                                label={{ value: 'CYCLES', position: 'insideBottomRight', offset: -5, fill: '#8f9196', fontSize: 10 }}
                            />
                            <YAxis
                                domain={[60, 100]}
                                stroke="#8f9196"
                                tick={{ fontSize: 10 }}
                                label={{ value: 'SOH %', angle: -90, position: 'insideLeft', fill: '#8f9196', fontSize: 10 }}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#8f9196', color: '#d9d7c5' }}
                                itemStyle={{ color: '#d9d7c5' }}
                            />
                            <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'EOL (80%)', fill: '#ef4444', fontSize: 10 }} />
                            <Line
                                type="monotone"
                                dataKey="managed"
                                stroke="#10b981"
                                strokeWidth={2}
                                dot={false}
                                name="Managed (Proactive)"
                            />
                            <Line
                                type="monotone"
                                dataKey="unmanaged"
                                stroke="#ef4444"
                                strokeWidth={2}
                                dot={false}
                                name="Unmanaged (Reactive)"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* AI Learning Curve (RMSE/MAE vs R2) */}
            <div className="group relative p-6 flex flex-col gap-4 hover-sketch min-h-[500px]">
                <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none" style={{ borderRadius: '25px 225px 25px 230px / 230px 25px 225px 25px' }}></div>
                <div className="flex justify-between items-start mb-4">
                    <Icon icon="lucide:brain-circuit" className="text-[#d9d7c5] w-5 h-5" />
                    <span className="text-xs text-[#8f9196] font-bold">AI_MODEL_TRAINING (ERROR vs ACCURACY)</span>
                </div>

                <div className="w-full h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={aiTrainingData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#8f9196" opacity={0.1} />
                            <XAxis
                                dataKey="epoch"
                                stroke="#8f9196"
                                tick={{ fontSize: 10 }}
                                label={{ value: 'EPOCHS', position: 'insideBottomRight', offset: -5, fill: '#8f9196', fontSize: 10 }}
                            />
                            {/* Left Axis: Error (RMSE, MAE) */}
                            <YAxis
                                yAxisId="left"
                                stroke="#ef4444"
                                tick={{ fontSize: 10, fill: '#ef4444' }}
                                label={{ value: 'ERROR (RMSE / MAE)', angle: -90, position: 'insideLeft', fill: '#ef4444', fontSize: 10 }}
                            />
                            {/* Right Axis: Accuracy (R2) */}
                            <YAxis
                                yAxisId="right"
                                orientation="right"
                                domain={[0, 1]}
                                stroke="#10b981"
                                tick={{ fontSize: 10, fill: '#10b981' }}
                                label={{ value: 'ACCURACY (R²)', angle: 90, position: 'insideRight', fill: '#10b981', fontSize: 10 }}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#8f9196', color: '#d9d7c5' }}
                                itemStyle={{ color: '#d9d7c5' }}
                            />
                            <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="rmse"
                                stroke="#ef4444"
                                strokeWidth={2}
                                dot={false}
                                name="RMSE"
                            />
                            <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="mae"
                                stroke="#f97316"
                                strokeWidth={2}
                                dot={false}
                                name="MAE"
                            />
                            <Line
                                yAxisId="right"
                                type="monotone"
                                dataKey="r2"
                                stroke="#10b981"
                                strokeWidth={2}
                                dot={false}
                                name="R² Score"
                            />
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>
                {/* Legend / Context */}
                <div className="mt-2 grid grid-cols-3 gap-4 text-[10px] text-[#8f9196] border-t border-[#8f9196]/10 pt-3">
                    <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-2 text-[#ef4444]">
                            <div className="w-3 h-1 bg-[#ef4444]"></div>
                            <span className="font-bold">RMSE</span>
                        </div>
                        <span className="opacity-70 leading-tight">Root Mean Square Error. Lower is better. Represents average prediction error magnitude.</span>
                    </div>
                    <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-2 text-[#f97316]">
                            <div className="w-3 h-1 bg-[#f97316]"></div>
                            <span className="font-bold">MAE</span>
                        </div>
                        <span className="opacity-70 leading-tight">Mean Absolute Error. Lower is better. Represents average absolute difference.</span>
                    </div>
                    <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-2 text-[#10b981]">
                            <div className="w-3 h-1 bg-[#10b981]"></div>
                            <span className="font-bold">R² Score</span>
                        </div>
                        <span className="opacity-70 leading-tight">Coefficient of Determination. Higher is better (max 1.0). Represents how well the model fits the data.</span>
                    </div>
                </div>
            </div>

            {/* Model Benchmark Heatmap */}
            <div className="group relative p-6 flex flex-col gap-4 hover-sketch">
                <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none" style={{ borderRadius: '20px 20px 225px 20px / 20px 225px 20px 225px' }}></div>
                <div className="flex justify-between items-start mb-4">
                    <Icon icon="lucide:bar-chart-4" className="text-[#d9d7c5] w-5 h-5" />
                    <span className="text-xs text-[#8f9196] font-bold">MODEL_BENCHMARK (BATCH 1.1)</span>
                </div>

                <div className="w-full overflow-x-auto">
                    <table className="w-full text-xs text-[#d9d7c5] border-collapse">
                        <thead>
                            <tr className="border-b border-[#8f9196]/20">
                                <th className="p-2 text-left font-bold text-[#8f9196]">Model</th>
                                <th className="p-2 text-center font-bold text-[#8f9196]">RMSE</th>
                                <th className="p-2 text-center font-bold text-[#8f9196]">MAE</th>
                                <th className="p-2 text-center font-bold text-[#8f9196]">R²</th>
                                <th className="p-2 text-center font-bold text-[#8f9196]">Time (s)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {[
                                { model: '1D-CNN', rmse: 0.0414, mae: 0.0369, r2: -0.7661, time: 0.013 },
                                { model: 'LSTM', rmse: 0.0245, mae: 0.0208, r2: 0.3830, time: 0.019 },
                                { model: 'ConvLSTM', rmse: 0.0124, mae: 0.0109, r2: 0.8420, time: 0.022 },
                                { model: 'Informer', rmse: 0.0289, mae: 0.0270, r2: 0.1400, time: 0.199 },
                                { model: 'BiLSTM', rmse: 0.0064, mae: 0.0054, r2: 0.9583, time: 0.018 },
                                { model: 'MBLSTM', rmse: 0.0286, mae: 0.0247, r2: 0.1583, time: 0.029 },
                                { model: 'iTransformer', rmse: 0.0056, mae: 0.0049, r2: 0.9712, time: 0.128 },
                                { model: 'MBLSTM+Informer', rmse: 0.0286, mae: 0.0247, r2: 0.1583, time: 0.196 },
                                { model: 'MBLSTM+iTransformer', rmse: 0.0037, mae: 0.0027, r2: 0.9876, time: 0.165, highlight: true },
                            ].map((row, idx) => (
                                <tr key={idx} className={`border-b border-[#8f9196]/10 ${row.highlight ? 'bg-[#10b981]/10' : ''}`}>
                                    <td className="p-2 font-mono">{row.model}</td>

                                    {/* RMSE: Lower is Green, Higher is Red */}
                                    <td className="p-2 text-center">
                                        <span className={`px-2 py-1 rounded ${row.rmse < 0.01 ? 'bg-[#10b981]/20 text-[#10b981]' :
                                            row.rmse < 0.03 ? 'bg-[#f59e0b]/20 text-[#f59e0b]' :
                                                'bg-[#ef4444]/20 text-[#ef4444]'
                                            }`}>
                                            {row.rmse.toFixed(4)}
                                        </span>
                                    </td>

                                    {/* MAE: Lower is Green */}
                                    <td className="p-2 text-center">
                                        <span className={`px-2 py-1 rounded ${row.mae < 0.01 ? 'bg-[#10b981]/20 text-[#10b981]' :
                                            row.mae < 0.025 ? 'bg-[#f59e0b]/20 text-[#f59e0b]' :
                                                'bg-[#ef4444]/20 text-[#ef4444]'
                                            }`}>
                                            {row.mae.toFixed(4)}
                                        </span>
                                    </td>

                                    {/* R2: Higher is Green */}
                                    <td className="p-2 text-center">
                                        <span className={`px-2 py-1 rounded ${row.r2 > 0.9 ? 'bg-[#10b981]/20 text-[#10b981]' :
                                            row.r2 > 0.5 ? 'bg-[#f59e0b]/20 text-[#f59e0b]' :
                                                'bg-[#ef4444]/20 text-[#ef4444]'
                                            }`}>
                                            {row.r2.toFixed(4)}
                                        </span>
                                    </td>

                                    {/* Time: Lower is Green */}
                                    <td className="p-2 text-center">
                                        <span className={`px-2 py-1 rounded ${row.time < 0.05 ? 'bg-[#10b981]/20 text-[#10b981]' :
                                            row.time < 0.15 ? 'bg-[#f59e0b]/20 text-[#f59e0b]' :
                                                'bg-[#ef4444]/20 text-[#ef4444]'
                                            }`}>
                                            {row.time.toFixed(3)}s
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Scenario Analysis Heatmap (Expanded Mock Data) */}
            <div className="group relative p-6 flex flex-col gap-4 hover-sketch">
                <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none" style={{ borderRadius: '20px 225px 20px 225px / 225px 20px 225px 20px' }}></div>
                <div className="flex justify-between items-start mb-4">
                    <Icon icon="lucide:grid" className="text-[#d9d7c5] w-5 h-5" />
                    <span className="text-xs text-[#8f9196] font-bold">SCENARIO_PERFORMANCE_MATRIX (RMSE HEATMAP)</span>
                </div>

                <div className="w-full overflow-x-auto">
                    <div className="min-w-[500px]">
                        {/* Header */}
                        <div className="grid grid-cols-5 gap-2 mb-2 text-xs font-bold text-[#8f9196] text-center">
                            <div className="text-left">SCENARIO</div>
                            <div>UNMANAGED</div>
                            <div>MANAGED</div>
                            <div>FAST CHARGE</div>
                            <div>COLD WEATHER</div>
                        </div>

                        {/* Baseline Row */}
                        <div className="grid grid-cols-5 gap-2 mb-2 items-center">
                            <div className="text-xs text-[#d9d7c5] font-mono">Baseline Arrhenius</div>
                            {/* Unmanaged */}
                            <div className="h-12 rounded bg-[#ef4444]/20 border border-[#ef4444]/40 flex flex-col items-center justify-center relative group/cell">
                                <span className="text-xs text-[#ef4444] font-bold">0.0452</span>
                                <span className="text-[9px] text-[#ef4444]/70">High Error</span>
                            </div>
                            {/* Managed */}
                            <div className="h-12 rounded bg-[#f59e0b]/20 border border-[#f59e0b]/40 flex flex-col items-center justify-center">
                                <span className="text-xs text-[#f59e0b] font-bold">0.0210</span>
                                <span className="text-[9px] text-[#f59e0b]/70">Med Error</span>
                            </div>
                            {/* Fast Charge (Mock) */}
                            <div className="h-12 rounded bg-[#ef4444]/30 border border-[#ef4444]/50 flex flex-col items-center justify-center">
                                <span className="text-xs text-[#ef4444] font-bold">0.0510</span>
                                <span className="text-[9px] text-[#ef4444]/70">Crit Error</span>
                            </div>
                            {/* Cold Weather (Mock) */}
                            <div className="h-12 rounded bg-[#f59e0b]/20 border border-[#f59e0b]/40 flex flex-col items-center justify-center">
                                <span className="text-xs text-[#f59e0b] font-bold">0.0350</span>
                                <span className="text-[9px] text-[#f59e0b]/70">Med Error</span>
                            </div>
                        </div>

                        {/* Proposed LSTM Row */}
                        <div className="grid grid-cols-5 gap-2 items-center">
                            <div className="text-xs text-[#10b981] font-mono font-bold">Proposed LSTM</div>
                            {/* Unmanaged */}
                            <div className="h-12 rounded bg-[#10b981]/20 border border-[#10b981]/40 flex flex-col items-center justify-center relative overflow-hidden">
                                <div className="absolute inset-y-0 left-0 w-1 bg-[#10b981]"></div>
                                <span className="text-xs text-[#10b981] font-bold">0.0013</span>
                                <span className="text-[9px] text-[#10b981]/70">97% Imp.</span>
                            </div>
                            {/* Managed */}
                            <div className="h-12 rounded bg-[#10b981]/30 border border-[#10b981]/50 flex flex-col items-center justify-center relative overflow-hidden">
                                <div className="absolute inset-y-0 left-0 w-1 bg-[#10b981]"></div>
                                <span className="text-xs text-[#10b981] font-bold">0.0003</span>
                                <span className="text-[9px] text-[#10b981]/70">99% Imp.</span>
                            </div>
                            {/* Fast Charge (Mock) */}
                            <div className="h-12 rounded bg-[#10b981]/20 border border-[#10b981]/40 flex flex-col items-center justify-center relative overflow-hidden">
                                <div className="absolute inset-y-0 left-0 w-1 bg-[#10b981]"></div>
                                <span className="text-xs text-[#10b981] font-bold">0.0021</span>
                                <span className="text-[9px] text-[#10b981]/70">96% Imp.</span>
                            </div>
                            {/* Cold Weather (Mock) */}
                            <div className="h-12 rounded bg-[#10b981]/20 border border-[#10b981]/40 flex flex-col items-center justify-center relative overflow-hidden">
                                <div className="absolute inset-y-0 left-0 w-1 bg-[#10b981]"></div>
                                <span className="text-xs text-[#10b981] font-bold">0.0018</span>
                                <span className="text-[9px] text-[#10b981]/70">95% Imp.</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Batch Analysis Visualization (Multi-Model) */}
            <div className="group relative p-6 flex flex-col gap-4 hover-sketch">
                <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none" style={{ borderRadius: '20px 225px 20px 225px / 225px 20px 225px 20px' }}></div>
                <div className="flex justify-between items-start mb-4">
                    <Icon icon="lucide:layers" className="text-[#d9d7c5] w-5 h-5" />
                    <span className="text-xs text-[#8f9196] font-bold">BATCH_ANALYSIS (MULTI-MODEL COMPARISON)</span>
                </div>

                <div className="grid grid-cols-1 gap-8">
                    {batchAnalysisData.map((batch) => (
                        <div key={batch.id} className="flex flex-col gap-2">
                            <div className="flex justify-between items-center px-2">
                                <span className="text-sm text-[#d9d7c5] font-bold">{batch.id}</span>
                                <span className="text-[10px] text-[#8f9196]">
                                    {batch.id === 'Batch 1.1' && 'Steady Decrease'}
                                    {batch.id === 'Batch 3.1' && 'Wavy Pattern'}
                                    {batch.id === 'Batch 4.1' && 'Nonlinear Decay'}
                                    {batch.id === 'Batch 5.1' && 'Irregular / Noisy'}
                                </span>
                            </div>
                            <div className="h-[350px] w-full border border-[#8f9196]/10 rounded bg-[#1a1a1a]/50 p-2 relative">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={batch.data}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#8f9196" opacity={0.1} />
                                        <XAxis
                                            dataKey="cycle"
                                            stroke="#8f9196"
                                            tick={{ fontSize: 10 }}
                                            interval={10}
                                            label={{ value: 'Cycle', position: 'insideBottom', offset: -5, fill: '#8f9196', fontSize: 10 }}
                                        />
                                        <YAxis
                                            domain={['auto', 'auto']}
                                            stroke="#8f9196"
                                            tick={{ fontSize: 10 }}
                                            width={35}
                                            label={{ value: 'SOH', angle: -90, position: 'insideLeft', fill: '#8f9196', fontSize: 10 }}
                                        />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#8f9196', color: '#d9d7c5', fontSize: '10px' }}
                                            itemStyle={{ color: '#d9d7c5' }}
                                            labelFormatter={(label) => `Cycle: ${label}`}
                                        />

                                        {/* Ground Truth */}
                                        <Line
                                            type="monotone"
                                            dataKey="actual"
                                            stroke="#d9d7c5"
                                            strokeWidth={2}
                                            strokeDasharray="5 5"
                                            dot={false}
                                            name="Ground Truth"
                                            isAnimationActive={false}
                                        />

                                        {/* All Models */}
                                        {batch.models.map((model) => (
                                            <Line
                                                key={model.name}
                                                type="monotone"
                                                dataKey={model.name}
                                                stroke={model.color}
                                                strokeWidth={model.isBest ? 2.5 : 1}
                                                dot={false}
                                                name={model.name}
                                                opacity={model.isBest ? 1 : 0.6}
                                                isAnimationActive={false}
                                            />
                                        ))}
                                    </LineChart>
                                </ResponsiveContainer>

                                {/* Mini Legend Overlay */}
                                <div className="absolute top-2 right-2 bg-[#1a1a1a]/90 border border-[#8f9196]/20 p-2 rounded text-[8px] flex flex-col gap-1 max-h-[100px] overflow-y-auto pointer-events-none">
                                    <div className="flex items-center gap-1">
                                        <div className="w-2 h-0.5 bg-[#d9d7c5] border-t border-dashed"></div>
                                        <span className="text-[#d9d7c5]">Ground Truth</span>
                                    </div>
                                    {batch.models.map(m => (
                                        <div key={m.name} className="flex items-center gap-1">
                                            <div className="w-2 h-0.5" style={{ backgroundColor: m.color }}></div>
                                            <span className={m.isBest ? 'text-[#d9d7c5] font-bold' : 'text-[#8f9196]'}>{m.name}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* LIME Explanation Visualization */}
            <div className="group relative p-6 flex flex-col gap-4 hover-sketch">
                <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500 pointer-events-none" style={{ borderRadius: '20px 225px 20px 225px / 225px 20px 225px 20px' }}></div>
                <div className="flex justify-between items-start mb-4">
                    <Icon icon="lucide:search" className="text-[#d9d7c5] w-5 h-5" />
                    <div className="flex flex-col">
                        <span className="text-xs text-[#8f9196] font-bold">LOCAL EXPLANATION (LIME)</span>
                        <span className="text-[10px] text-[#ef4444]">Anomaly Detected: Batch 5.1 @ Cycle 42</span>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {limeData.map((battery) => (
                        <div key={battery.id} className="flex flex-col gap-2">
                            <span className="text-xs text-[#d9d7c5] font-bold text-center">({battery.id}) Local Explanation</span>
                            <div className="h-[250px] border border-[#8f9196]/10 rounded bg-[#1a1a1a]/50 p-2">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart
                                        layout="vertical"
                                        data={battery.data}
                                        margin={{ top: 5, right: 10, left: 10, bottom: 5 }}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" stroke="#8f9196" opacity={0.1} horizontal={false} />
                                        <XAxis type="number" stroke="#8f9196" tick={{ fontSize: 8 }} />
                                        <YAxis
                                            dataKey="feature"
                                            type="category"
                                            stroke="#8f9196"
                                            tick={{ fontSize: 8 }}
                                            width={90}
                                        />
                                        <Tooltip
                                            cursor={{ fill: '#8f9196', opacity: 0.1 }}
                                            contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#8f9196', color: '#d9d7c5', fontSize: '10px' }}
                                            itemStyle={{ color: '#d9d7c5' }}
                                        />
                                        <ReferenceLine x={0} stroke="#8f9196" />
                                        <Bar dataKey="value" name="Impact">
                                            {battery.data.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#10b981' : '#ef4444'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
