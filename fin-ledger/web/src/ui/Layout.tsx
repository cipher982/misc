import React from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { Bars3Icon } from '@heroicons/react/24/outline'

export function Layout() {
  const [open, setOpen] = React.useState(false)
  const link = (to: string, label: string) => (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-3 py-2 rounded-md text-sm font-medium ${isActive ? 'bg-slate-800 text-white' : 'text-slate-300 hover:text-white hover:bg-slate-800'}`
      }
      onClick={() => setOpen(false)}
    >
      {label}
    </NavLink>
  )

  return (
    <div className="min-h-screen grid grid-rows-[auto,1fr]">
      <header className="border-b border-slate-800 bg-slate-950/70 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button className="md:hidden p-2 rounded bg-slate-900 border border-slate-800" onClick={() => setOpen(!open)}>
              <Bars3Icon className="h-5 w-5 text-slate-300" />
            </button>
            <div className="text-lg font-semibold">Fin Ledger</div>
          </div>
          <nav className="hidden md:flex items-center gap-2">
            {link('/', 'Dashboard')}
            {link('/transactions', 'Transactions')}
            {link('/fees', 'Fees')}
            {link('/spend', 'Spend')}
            {link('/statements', 'Statements')}
            {link('/documents', 'Documents')}
            {link('/benefits', 'Benefits')}
          </nav>
        </div>
        {open && (
          <div className="md:hidden border-t border-slate-800 px-4 py-2 flex flex-col gap-2">
            {link('/', 'Dashboard')}
            {link('/transactions', 'Transactions')}
            {link('/fees', 'Fees')}
            {link('/spend', 'Spend')}
            {link('/statements', 'Statements')}
            {link('/documents', 'Documents')}
            {link('/benefits', 'Benefits')}
          </div>
        )}
      </header>
      <main className="mx-auto max-w-7xl w-full p-4">
        <Outlet />
      </main>
    </div>
  )
}
