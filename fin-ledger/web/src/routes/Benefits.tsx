import React from 'react'
import { getBenefits } from '../lib/api'

export function Benefits() {
  const [year, setYear] = React.useState(new Date().getFullYear())
  const [data, setData] = React.useState<{product:string;issuer:string;year:number;credits:any[]} | null>(null)

  async function run() {
    setData(await getBenefits('venture_x', year))
  }

  React.useEffect(() => { run() }, [])

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">Benefits usage</h2>
        <div className="flex items-center gap-2">
          <input className="input w-28" value={year} onChange={e=>setYear(parseInt(e.target.value||'0')||new Date().getFullYear())} />
          <button className="btn" onClick={run}>Refresh</button>
        </div>
      </div>
      <div className="grid gap-3">
        {data?.credits?.map((c) => (
          <div key={c.key} className="p-4 rounded bg-slate-900 border border-slate-800">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">{c.name}</div>
                <div className="text-slate-400 text-sm">Window: {c.window}</div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-semibold">${c.used.toFixed(2)} / ${c.cap.toFixed(2)}</div>
                <div className="text-slate-400 text-sm">Remaining: ${c.remaining.toFixed(2)}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
