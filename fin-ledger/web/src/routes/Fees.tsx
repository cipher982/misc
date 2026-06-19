import React from 'react'
import { getFees } from '../lib/api'

export function Fees() {
  const [since, setSince] = React.useState(() => new Date(Date.now()-1000*60*60*24*90).toISOString().slice(0,10))
  const [data, setData] = React.useState<{count:number; total:number} | null>(null)

  React.useEffect(() => { getFees(since).then(setData) }, [since])

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">Fees</h2>
        <input className="input" value={since} onChange={e=>setSince(e.target.value)} />
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 rounded bg-slate-900 border border-slate-800">
          <div className="text-slate-400 text-sm">Count</div>
          <div className="text-2xl font-semibold">{data?.count ?? 0}</div>
        </div>
        <div className="p-4 rounded bg-slate-900 border border-slate-800">
          <div className="text-slate-400 text-sm">Total</div>
          <div className="text-2xl font-semibold">${data?.total.toFixed(2) ?? '0.00'}</div>
        </div>
      </div>
    </div>
  )
}
