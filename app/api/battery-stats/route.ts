import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import util from 'util';

const execPromise = util.promisify(exec);

export async function GET() {
    try {
        const { stdout } = await execPromise('ioreg -r -n AppleSmartBattery -w0');

        // Helper regex extraction
        const extractInt = (key: string) => {
            const match = stdout.match(new RegExp(`"${key}" = (\\d+)`));
            return match ? parseInt(match[1], 10) : null;
        };

        const extractStr = (key: string) => {
            const match = stdout.match(new RegExp(`"${key}" = (.+)`));
            return match ? match[1].trim() : null;
        };

        const stats = {
            cycleCount: extractInt("CycleCount") || 0,
            maxCapacity: extractInt("AppleRawMaxCapacity") || 0,
            designCapacity: extractInt("DesignCapacity") || 0,
            temperature: (extractInt("Temperature") || 0) / 100.0, // 0.01 C -> C
            voltage: (extractInt("Voltage") || 0) / 1000.0, // mV -> V
            isCharging: extractStr("IsCharging") === "Yes",
            soh: 0
        };

        if (stats.maxCapacity && stats.designCapacity) {
            stats.soh = (stats.maxCapacity / stats.designCapacity) * 100.0;
        }

        return NextResponse.json(stats);
    } catch (error) {
        console.error("Error fetching battery stats:", error);
        return NextResponse.json({ error: "Failed to fetch battery stats" }, { status: 500 });
    }
}
