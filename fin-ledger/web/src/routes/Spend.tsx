import React from 'react'
import { getSpend } from '../lib/api'

export function Spend() {
  const [merchant, setMerchant] = React.useState('Amazon')
  const [since, setSince] = React.useState(() => `${new Date().getFullYear()}-01-01`)
  const [data, setData] = React.useState<{total:number;count:number;last_date:string|null} | null>(null)

  async function run() {
    setData(await getSpend(merchant, since))
  }

  React.useEffect(() => { run() }, [])

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">Merchant spend</h2>
        <div className="flex items-center gap-2">
          <input className="input" placeholder="Merchant" value={merchant} onChange={e=>setMerchant(e.target.value)} />
          <input className="input" value={since} onChange={e=>setSince(e.target.value)} />
          <button className="btn" onClick={run}>Run</button>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <div className="p-4 rounded bg-slate-900 border border-slate-800">
          <div className="text-slate-400 text-sm">Total</div>
          <div className="text-2xl font-semibold">${data?.total?.toFixed(2) ?? '0.00'}</div>
        </div>
        <div className="p-4 rounded bg-slate-900 border border-slate-800">
          <div className="text-slate-400 text-sm">Transactions</div>
          <div className="text-2xl font-semibold">{data?.count ?? 0}</div>
        </div>
        <div className="p-4 rounded bg-slate-900 border border-slate-800">
          <div className="text-slate-400 text-sm">Last date</div>
          <div className="text-2xl font-semibold">{data?.last_date ?? '—'}</div>
        </div>
      </div>
    </div>
  )
}
