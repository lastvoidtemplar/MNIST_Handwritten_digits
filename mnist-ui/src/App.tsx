// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
// import './App.css'
import DigitCanvas from './components/DigitCanvas'
export const PIXEL_SIZE = 15
function App() {
  return (
    <div className='w-screen h-screen flex justify-center items-center'>
       <DigitCanvas width={28*PIXEL_SIZE} height={28*PIXEL_SIZE}/>
    </div>
  
  )
}

export default App
