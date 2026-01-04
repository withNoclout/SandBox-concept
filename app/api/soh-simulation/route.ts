import { NextResponse } from 'next/server';

interface SimulationRequest {
    avgTemp: number;
    chargeLimit: number;
    dod: number;
    cRate: number;
    initialSoh: number;
    initialCycles: number;
}

export async function POST(request: Request) {
    try {
        const body: SimulationRequest = await request.json();
        const { avgTemp, chargeLimit, dod, cRate, initialSoh, initialCycles } = body;

        const cyclesToSimulate = 1000;
        const history = [];
        let currentSoh = initialSoh;

        // --- STRESS FACTORS ---
        const tempStress = 1 + Math.max(0, (avgTemp - 25) * 0.05);
        const voltageStress = chargeLimit > 80 ? 1.5 : 1.0;
        const dodStress = Math.pow(dod / 0.5, 1.5);
        const cRateStress = Math.pow(cRate, 1.2);
        const baseDegradation = 0.02;

        // --- PHYSICS MODEL CONSTANTS ---
        // SEI Growth Factor (Square Root Time Dependence)
        const SEI_FACTOR = 0.5;

        // Knee Onset Threshold (Accelerated degradation start)
        const KNEE_THRESHOLD = 80.0;

        // Cycle-based aging accumulation
        let effectiveAge = 0;

        for (let i = 0; i < cyclesToSimulate; i++) {
            // 1. Calculate Stress Multiplier for this cycle
            const stressMultiplier = tempStress * voltageStress * dodStress * cRateStress;

            // 2. Accumulate "Effective Age" (Time/Stress integral)
            // Instead of linear subtraction, we accumulate stress-weighted time
            effectiveAge += stressMultiplier;

            // 3. Calculate Base Degradation (SEI Layer Growth ~ sqrt(t))
            // Delta SOH = k * sqrt(t)
            // We calculate the incremental drop for this step
            const degradationSEI = SEI_FACTOR * (Math.sqrt(effectiveAge) - Math.sqrt(effectiveAge - stressMultiplier));

            // 4. Knee Effect (Lithium Plating / Active Material Loss)
            // If SOH drops below threshold, add exponential penalty
            let degradationKnee = 0;
            if (currentSoh < KNEE_THRESHOLD) {
                const depthBelowKnee = KNEE_THRESHOLD - currentSoh;
                degradationKnee = 0.001 * Math.pow(depthBelowKnee, 2); // Quadratic acceleration
            }

            // 5. Total Daily Degradation
            let totalDegradation = degradationSEI + degradationKnee;

            // 6. Stochastic Noise
            const u1 = Math.random();
            const u2 = Math.random();
            const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            const noise = z * (0.1 * totalDegradation); // Noise proportional to degradation

            totalDegradation = Math.max(0, totalDegradation + noise);
            currentSoh -= totalDegradation;

            history.push({
                cycle: initialCycles + i + 1,
                soh: currentSoh
            });
        }

        return NextResponse.json(history);
    } catch (error) {
        console.error("Error running simulation:", error);
        return NextResponse.json({ error: "Simulation failed" }, { status: 500 });
    }
}
